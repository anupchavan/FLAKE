"""FLAKE -- Federated Learning with Aggregated Knowledge-Distillation Exchange.

Decentralised, peer-to-peer federated learning on CIFAR-10.  There is no
central server and no shared global model: every client talks only to its
peers over TCP.

Round r on client i:

  1.  Pull the latest weights of every other client (each client runs a small
      TCP listener that serves its most recent weights on demand).
  2.  Locally FedAvg (own weights + every responding peer), sample-weighted.
      Call the result ``w_agg``.
  3.  Freeze a copy of ``w_agg`` as this round's *teacher*.
  4.  Reset the student to ``w_agg`` and train for ``E`` SGD epochs with the
      knowledge-distillation loss ``L = CE + mu * KL(student || teacher)``
      (the true class column is masked out of the soft-label term so the
      teacher can't contradict the hard label).
  5.  Publish the new weights for peers to pull next round.

Each client starts from the same seeded initial weights (so round 0's
aggregation is meaningful), then runs a one-epoch cross-entropy warm-up on
its own data.  The KD weight ``mu`` is linearly ramped from 0 to ``KD_MU``
over the first few rounds so the student is not dragged toward a weak
early-round teacher.

Input file (``flake_input.txt`` by default; override with ``FLAKE_INPUT``)::

    N M
    CURRENT_MACHINE_IP
    client_ip_1,client_ip_2,...,client_ip_N
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import random
import socket
import struct
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# =============================================================================
# Reproducibility -- one shared seed so both framework scripts get the same
# Dirichlet split and the same initial weights.
# =============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# Hyperparameters (all overridable via FL_* environment variables so the
# compare.py harness can align the two frameworks run-to-run).
# =============================================================================
BATCH_SIZE = int(os.environ.get("FL_BATCH_SIZE", "32"))
EPOCHS_PER_ROUND = int(os.environ.get("FL_EPOCHS_PER_ROUND", "1"))
TOTAL_ROUNDS = int(os.environ.get("FL_ROUNDS", "100"))
DIRICHLET_ALPHA = float(os.environ.get("FL_DIRICHLET_ALPHA", "0.5"))
FIXED_DATA_PER_CLIENT = 5000
DEVICE = torch.device("cpu")
LOCAL_LR = 0.01

# KD loss hyperparameters. tau sharpens the softmax; mu scales the KL term
# relative to the cross-entropy term.
KD_TEMPERATURE = float(os.environ.get("FL_FLAKE_TAU", "1.0"))
KD_MU = float(os.environ.get("FL_FLAKE_MU", "1.5"))

# Stability knobs for the multi-threaded localhost setup:
#   WARMUP_EPOCHS   -- CE-only epochs before the first KD round, so the
#                      round-0 teacher is not a random network.
#   KD_WARMUP_ROUNDS -- linearly ramp mu from 0 to KD_MU over this many
#                       rounds so the student is not pulled toward a weak
#                       early-round teacher.
KD_WARMUP_EPOCHS = int(os.environ.get("FL_FLAKE_WARMUP_EPOCHS", "1"))
KD_WARMUP_ROUNDS = int(os.environ.get("FL_FLAKE_KD_WARMUP_ROUNDS", "3"))

# Networking.
CONNECT_TIMEOUT = float(os.environ.get("FED_CONNECT_TIMEOUT", "60"))
PEER_PULL_TIMEOUT = float(os.environ.get("FED_PEER_PULL_TIMEOUT", "60"))
SERVER_BACKLOG = int(os.environ.get("FED_SERVER_BACKLOG", "128"))
# Dedicated port range so FLAKE and Layer Sharing can coexist on one host.
CLIENT_BASE_PORT = int(os.environ.get("FLAKE_BASE_PORT", "8700"))


# =============================================================================
# Logging (console + file handler added in main() once model name is known)
# =============================================================================
logger = logging.getLogger("flake")
logger.setLevel(logging.INFO)
_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
logger.addHandler(_console_handler)


class _LoggerWriter:
    """Tiny stdout replacement that forwards print() calls through the logger."""

    def __init__(self, lg, level):
        self.lg = lg
        self.level = level

    def write(self, message):
        if message.strip():
            self.lg.log(self.level, message.strip())

    def flush(self):
        pass


sys.stdout = _LoggerWriter(logger, logging.INFO)


# =============================================================================
# Input parsing
# =============================================================================
def parse_input_file(path: str = "flake_input.txt"):
    """Read ``N M``, the current machine's IP, and the list of N client IPs
    from the FLAKE input file. Returns ``(N, M, current_ip, ips)`` or all
    ``None`` on a parse error."""
    try:
        with open(path, "r") as fh:
            lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
        if len(lines) < 3:
            raise ValueError("Input file must have at least 3 non-empty lines.")
        n, m = map(int, lines[0].split())
        current_machine_ip = lines[1]
        client_ips = [ip.strip() for ip in lines[2].split(",")]
        if len(client_ips) != n:
            raise ValueError(f"Expected {n} client IPs on line 3, got {len(client_ips)}.")
        return n, m, current_machine_ip, client_ips
    except FileNotFoundError:
        print(f"Input file '{path}' not found.")
    except ValueError as ve:
        print(f"Error parsing input file: {ve}")
    except Exception as e:
        print(f"Unexpected error parsing input file: {e}")
    return None, None, None, None


_INPUT_PATH = os.environ.get("FLAKE_INPUT", "flake_input.txt")
NUM_CLIENTS, NUM_MACHINES, CURRENT_MACHINE_IP, ips = parse_input_file(_INPUT_PATH)
if NUM_CLIENTS is None:
    print("Failed to parse the input file. Exiting.")
    sys.exit(1)


# =============================================================================
# Networking utilities: length-prefixed pickle messages over plain TCP.
# =============================================================================
def send_message(conn, message):
    """Pickle ``message`` and send it with a 4-byte big-endian length prefix."""
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(conn, nbytes: int):
    """Read exactly ``nbytes`` from ``conn``; returns ``None`` on early close."""
    data = b""
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def receive_message(conn):
    """Inverse of :func:`send_message`: read the length prefix then the body."""
    hdr = _recv_exact(conn, 4)
    if not hdr:
        return None
    msg_len = struct.unpack("!I", hdr)[0]
    body = _recv_exact(conn, msg_len)
    if body is None:
        return None
    return pickle.loads(body)


def _new_socket():
    """Return a fresh TCP socket with enlarged send/receive buffers."""
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
        sk.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
    except Exception:
        pass
    return sk


# =============================================================================
# Dataset -- CIFAR-10 with the same seeded Dirichlet non-IID partition used by
# layer_sharing.py, so both frameworks see identical client-local data.
# =============================================================================
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=_transform)
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=_transform)


def create_dirichlet_non_iid_splits_fixed(dataset, num_clients, alpha=0.5, fixed_per_client=5000):
    """Split ``dataset`` into ``num_clients`` non-IID shards using a
    class-aware Dirichlet(alpha) distribution, then truncate / up-sample each
    shard to exactly ``fixed_per_client`` examples. Lower alpha -> more skew.
    """
    num_classes = 10
    class_indices = {c: np.where(np.array(dataset.targets) == c)[0] for c in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}
    for c, idxs in class_indices.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idxs)).astype(int)
        start = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(idxs[start:start + count])
            start += count
    final = {}
    for cid, idxs in client_indices.items():
        np.random.shuffle(idxs)
        if len(idxs) > fixed_per_client:
            final[cid] = list(idxs[:fixed_per_client])
        else:
            final[cid] = list(np.random.choice(idxs, fixed_per_client, replace=True))
    return [torch.utils.data.Subset(dataset, final[i]) for i in range(num_clients)]


client_data = create_dirichlet_non_iid_splits_fixed(
    train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, fixed_per_client=FIXED_DATA_PER_CLIENT
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)


# =============================================================================
# Model zoo. FLAKE is architecture-agnostic: KD is applied at the logit
# level and FedAvg at the state-dict level, so any CIFAR-10 classifier with
# identical architecture across clients is a drop-in.
# =============================================================================
class PaperCNN(nn.Module):
    """Two 5x5 conv layers + 2x2 max-pool + two fully-connected layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN(nn.Module):
    """Small 3x3 CNN baseline shared with Layer Sharing for a like-for-like comparison."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN10(nn.Module):
    """10-layer CNN: 4 conv blocks (64-128-256-512) + 2 FC, ~4.7M params."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def _make_vgg_layers(cfg):
    """Build the convolutional trunk of a VGG network from a channel/'M' config."""
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            layers.extend([
                nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                nn.BatchNorm2d(v),
                nn.ReLU(True),
            ])
            in_channels = v
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """CIFAR-sized VGG with batch norm. Five 2x2 pools reduce 32x32 to 1x1."""

    def __init__(self, features, num_classes: int = 10):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, 512), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def VGG11BN():
    """Constructor for VGG-11 with batch norm (CIFAR-sized)."""
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def VGG13BN():
    """Constructor for VGG-13 with batch norm (CIFAR-sized)."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def VGG16BN():
    """Constructor for VGG-16 with batch norm (CIFAR-sized)."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
           512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


class _CifarBasicBlock(nn.Module):
    """Two 3x3 convs + BN + ReLU with a projection shortcut when dims change."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class _CifarResNet(nn.Module):
    """CIFAR ResNet (He et al. 2015). Depth = 6n + 2; n=3 gives ResNet-20."""

    def __init__(self, n: int = 3, num_classes: int = 10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, n, stride):
        """Stack ``n`` basic blocks. The first one may down-sample via ``stride``."""
        layers = [_CifarBasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(_CifarBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)


def ResNet20():
    """ResNet-20 for CIFAR (20 layers, ~0.27M params)."""
    return _CifarResNet(n=3)


def build_model(choice: int) -> nn.Module:
    """Instantiate the model selected by ``--model`` (see MODEL_NAME_MAP)."""
    if choice == 1:
        return PaperCNN()
    if choice == 2:
        return SimpleCNN()
    if choice == 3:
        return SimpleCNN10()
    if choice == 4:
        return VGG11BN()
    if choice == 5:
        return VGG13BN()
    if choice == 6:
        return VGG16BN()
    if choice == 7:
        return ResNet20()
    raise ValueError(
        "Model choice must be 1..7 "
        "(1=PaperCNN, 2=SimpleCNN, 3=SimpleCNN10, "
        "4=VGG11-BN, 5=VGG13-BN, 6=VGG16-BN, 7=ResNet-20-CIFAR)."
    )


MODEL_NAME_MAP = {
    1: "PaperCNN",
    2: "SimpleCNN",
    3: "SimpleCNN10",
    4: "VGG11-BN",
    5: "VGG13-BN",
    6: "VGG16-BN",
    7: "ResNet-20-CIFAR",
}


# =============================================================================
# State-dict helpers: pickle-friendly numpy dicts cross the wire between peers.
# =============================================================================
def state_dict_to_numpy(model: nn.Module):
    """Snapshot a model's state_dict as pure numpy (safe to pickle and send)."""
    return {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}


def numpy_to_state_dict_torch(state_np):
    """Convert a numpy state_dict back to torch tensors on ``DEVICE``."""
    return {k: torch.tensor(v).to(DEVICE) for k, v in state_np.items()}


def fedavg_aggregate(state_dicts, weights):
    """Sample-weighted FedAvg over a list of numpy state_dicts.

    Floating-point tensors are averaged in float64 then cast back; integer
    buffers (e.g. BatchNorm's ``num_batches_tracked``) are copied from the
    first participant unchanged.
    """
    total = float(sum(weights))
    if total <= 0 or not state_dicts:
        return None
    norm_w = [w / total for w in weights]
    out = {}
    for key in state_dicts[0].keys():
        ref = state_dicts[0][key]
        if np.issubdtype(ref.dtype, np.floating):
            acc = np.zeros_like(ref, dtype=np.float64)
            for sd, w in zip(state_dicts, norm_w):
                acc += sd[key].astype(np.float64) * w
            out[key] = acc.astype(ref.dtype)
        else:
            out[key] = ref.copy()
    return out


def compute_accuracy_and_f1(model, loader, num_classes: int = 10):
    """Evaluate test accuracy (%) and macro-F1 on ``loader``."""
    model.eval()
    tp = np.zeros(num_classes, dtype=np.int64)
    fp = np.zeros(num_classes, dtype=np.int64)
    fn = np.zeros(num_classes, dtype=np.int64)
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            preds = model(data).argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
            for c in range(num_classes):
                pc = preds == c
                tc = target == c
                tp[c] += int((pc & tc).sum().item())
                fp[c] += int((pc & ~tc).sum().item())
                fn[c] += int((~pc & tc).sum().item())
    acc = 100.0 * correct / max(1, total)
    f1_per_class = []
    for c in range(num_classes):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1_per_class.append((2 * tp[c]) / denom if denom > 0 else 0.0)
    return acc, float(np.mean(f1_per_class))


# =============================================================================
# Knowledge-distillation loss with the true-class column masked out of the
# soft-label term (the hard CE term handles that column).
# =============================================================================
def kd_local_loss(student_logits, teacher_logits, targets, tau: float, mu: float):
    """Compute ``CE + mu * KL(student || teacher)``.

    The true-class logit is masked to -inf on both sides before the softmax
    is taken so the KL only carries soft information about the *wrong*
    classes; the CE term already supervises the true class directly.
    """
    ce = F.cross_entropy(student_logits, targets)

    keep_mask = torch.ones_like(student_logits, dtype=torch.bool)
    keep_mask.scatter_(1, targets.view(-1, 1), False)
    neg_inf = torch.finfo(student_logits.dtype).min

    t_masked = teacher_logits.masked_fill(~keep_mask, neg_inf) / tau
    s_masked = student_logits.masked_fill(~keep_mask, neg_inf) / tau

    log_q_s = F.log_softmax(s_masked, dim=1)
    q_t = F.softmax(t_masked, dim=1)
    kl = F.kl_div(log_q_s, q_t, reduction="batchmean")
    return ce + mu * kl, ce.detach().item(), kl.detach().item()


# =============================================================================
# Per-client TCP listener: replies to ``weights_request`` messages with the
# client's latest published weights and sample count.
# =============================================================================
def tcp_server(client_id: int, shared: dict, lock: threading.Lock,
               stop_event: threading.Event):
    """Serve this client's most recent weights to any peer that asks for them.

    Accepts incoming connections until ``stop_event`` is set.  Each
    connection expects a single ``weights_request`` message and is answered
    with the client's current ``shared["weights"]`` snapshot.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", CLIENT_BASE_PORT + client_id))
    server.listen(SERVER_BACKLOG)
    server.settimeout(1.0)
    logger.info(f"[client {client_id}] listening on :{CLIENT_BASE_PORT + client_id}")

    def _handle(conn):
        try:
            conn.settimeout(CONNECT_TIMEOUT)
            msg = receive_message(conn)
            if not msg or msg.get("type") != "weights_request":
                return
            with lock:
                weights = shared["weights"]
                n_samples = shared["n_samples"]
                round_id = shared["round"]
            send_message(conn, {
                "type": "weights_response",
                "client_id": client_id,
                "round": round_id,
                "n_samples": n_samples,
                "weights": weights,
            })
        except Exception as e:
            logger.warning(f"[client {client_id}] listener error: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    try:
        while not stop_event.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            threading.Thread(target=_handle, args=(conn,), daemon=True).start()
    finally:
        try:
            server.close()
        except Exception:
            pass
        logger.info(f"[client {client_id}] listener stopped")


_PEER_PULL_ATTEMPTS = int(os.environ.get("FL_PEER_PULL_ATTEMPTS", "30"))
_PEER_PULL_DELAY = float(os.environ.get("FL_PEER_PULL_DELAY", "1.0"))


def request_peer_weights(requester_id: int, target_id: int, target_ip: str,
                         current_round: int,
                         attempts: int = _PEER_PULL_ATTEMPTS,
                         delay: float = _PEER_PULL_DELAY):
    """Pull the current weights of peer ``target_id`` over TCP.

    Retries up to ``attempts`` times with ``delay`` seconds between attempts
    so a peer that is still booting its listener does not get excluded
    immediately.  Returns the response dict or ``None`` on persistent failure.
    """
    for _ in range(attempts):
        sock = _new_socket()
        sock.settimeout(PEER_PULL_TIMEOUT)
        try:
            sock.connect((target_ip, CLIENT_BASE_PORT + target_id))
            send_message(sock, {
                "type": "weights_request",
                "requester_id": requester_id,
                "round": current_round,
            })
            return receive_message(sock)
        except (ConnectionRefusedError, OSError):
            time.sleep(delay)
        finally:
            try:
                sock.close()
            except Exception:
                pass
    logger.warning(
        f"[client {requester_id}] could not pull weights from peer {target_id} "
        f"({target_ip}:{CLIENT_BASE_PORT + target_id}) after {attempts} attempts"
    )
    return None


# =============================================================================
# Per-client training loop (runs in its own thread; NUM_CLIENTS of these per host).
# =============================================================================
def client_logic(client_id: int, model_choice: int, results_store: dict,
                 results_lock: threading.Lock,
                 initial_state_np: dict | None = None,
                 warmup_barrier: threading.Barrier | None = None):
    """Implements the full FLAKE round loop for one client.

    Boots a peer listener thread, runs a one-epoch CE warm-up, then for every
    federated round pulls peer weights, locally FedAvgs them, trains one (or
    more) KD epochs against the resulting teacher, and publishes the new
    weights. Accuracy + F1 are evaluated at the end of every round.  Results
    are written to ``results_store[client_id]``.
    """
    train_loader = torch.utils.data.DataLoader(
        client_data[client_id], batch_size=BATCH_SIZE, shuffle=True
    )
    n_local = len(client_data[client_id])

    model = build_model(model_choice).to(DEVICE)

    # All clients start from the SAME seeded initial weights, so the round-0
    # FedAvg does not collapse to the mean of unrelated random networks.
    if initial_state_np is not None:
        model.load_state_dict(numpy_to_state_dict_torch(initial_state_np),
                              strict=True)

    # Shared state served to peers via the listener.
    #   round = -2  -> pre-warmup (initial weights)
    #   round = -1  -> warmup complete
    #   round >= 0  -> regular KD round
    shared = {
        "weights": state_dict_to_numpy(model),
        "n_samples": n_local,
        "round": -2,
    }
    state_lock = threading.Lock()
    stop_event = threading.Event()

    server_thread = threading.Thread(
        target=tcp_server,
        args=(client_id, shared, state_lock, stop_event),
        daemon=True,
    )
    server_thread.start()
    time.sleep(1.0)  # let the listener bind before peers start pulling

    history = []
    best_acc, best_round = 0.0, -1
    final_acc, final_f1 = 0.0, 0.0
    train_secs = 0.0
    comm_secs = 0.0
    t_client_0 = time.time()

    # ---- 0. Warm-up: one CE-only epoch so round-0's teacher is non-random. ----
    if KD_WARMUP_EPOCHS > 0:
        warmup_opt = optim.SGD(model.parameters(), lr=LOCAL_LR, momentum=0.9)
        model.train()
        t_warm0 = time.perf_counter()
        for _ in range(KD_WARMUP_EPOCHS):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                warmup_opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                warmup_opt.step()
        train_secs += time.perf_counter() - t_warm0
        with state_lock:
            shared["weights"] = state_dict_to_numpy(model)
            shared["round"] = -1
        print(f"[client {client_id}] warmup done "
              f"({KD_WARMUP_EPOCHS} CE epoch(s) on local data)")

    # All co-hosted clients wait here so nobody pulls pre-warmup weights.
    if warmup_barrier is not None:
        try:
            warmup_barrier.wait(timeout=120.0)
        except threading.BrokenBarrierError:
            logger.warning(
                f"[client {client_id}] warmup barrier broken; proceeding"
            )

    for r in range(TOTAL_ROUNDS):
        if stop_event.is_set():
            break

        # ---- 1. Pull current weights from every peer (in parallel). ----
        peer_states = {}
        peer_lock = threading.Lock()

        def _pull(peer_id, peer_ip):
            resp = request_peer_weights(client_id, peer_id, peer_ip, r)
            if resp is not None and resp.get("type") == "weights_response":
                with peer_lock:
                    peer_states[peer_id] = (resp["weights"], int(resp["n_samples"]))

        t_comm0 = time.perf_counter()
        threads = []
        for peer_id in range(NUM_CLIENTS):
            if peer_id == client_id:
                continue
            th = threading.Thread(target=_pull, args=(peer_id, ips[peer_id]),
                                  daemon=True)
            th.start()
            threads.append(th)
        for th in threads:
            th.join()
        comm_secs += time.perf_counter() - t_comm0

        # ---- 2. Local FedAvg (own + every peer that responded). ----
        with state_lock:
            own_weights = shared["weights"]
        all_states = [own_weights] + [s for s, _ in peer_states.values()]
        all_n = [n_local] + [n for _, n in peer_states.values()]
        aggregated_state = fedavg_aggregate(all_states, all_n)
        if aggregated_state is None:
            logger.warning(f"[client {client_id}] round {r}: aggregation failed")
            continue

        # ---- 3. Frozen aggregated peer model = this round's teacher. ----
        teacher = build_model(model_choice).to(DEVICE)
        teacher.load_state_dict(numpy_to_state_dict_torch(aggregated_state), strict=True)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        # ---- 4. Student starts from the consensus, trains with KD loss. ----
        model.load_state_dict(numpy_to_state_dict_torch(aggregated_state), strict=True)
        optimizer = optim.SGD(model.parameters(), lr=LOCAL_LR, momentum=0.9)

        # Linear KD-weight ramp: early rounds use less KD because the peer
        # teacher is still converging toward a useful distribution.
        if KD_WARMUP_ROUNDS > 0:
            mu_r = KD_MU * min(1.0, (r + 1) / float(KD_WARMUP_ROUNDS))
        else:
            mu_r = KD_MU

        t_train0 = time.perf_counter()
        model.train()
        for _epoch in range(EPOCHS_PER_ROUND):
            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                with torch.no_grad():
                    t_logits = teacher(x)
                s_logits = model(x)
                loss, _, _ = kd_local_loss(s_logits, t_logits, y,
                                           tau=KD_TEMPERATURE, mu=mu_r)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        train_secs += time.perf_counter() - t_train0

        # ---- 5. Publish updated weights for peers to pull next round. ----
        new_weights = state_dict_to_numpy(model)
        with state_lock:
            shared["weights"] = new_weights
            shared["round"] = r

        # ---- 6. Evaluate. ----
        acc, f1 = compute_accuracy_and_f1(model, test_loader)
        final_acc, final_f1 = float(acc), float(f1)
        if acc > best_acc:
            best_acc, best_round = float(acc), int(r)
        history.append({
            "round": r,
            "acc": float(acc),
            "f1": float(f1),
            "peers_responded": len(peer_states),
        })
        print(
            f"[client {client_id}] round {r}: acc={acc:.2f}% F1={f1:.4f} "
            f"peers={len(peer_states)}/{NUM_CLIENTS - 1}"
        )

    total_s = time.time() - t_client_0

    # Keep the listener alive briefly so straggling peers can still finish
    # their last pull before we tear it down.
    time.sleep(2.0)
    stop_event.set()
    server_thread.join(timeout=3.0)

    with results_lock:
        results_store[client_id] = {
            "final_acc": final_acc,
            "final_f1": final_f1,
            "best_acc": best_acc,
            "best_round": best_round,
            "last_round": int(history[-1]["round"]) if history else -1,
            "training_s": float(train_secs),
            "comm_s": float(comm_secs),
            "total_s": float(total_s),
            "history": history,
        }


# =============================================================================
# Main entry point
# =============================================================================
def main():
    """Parse CLI args, spawn one thread per locally-hosted client, and write a
    summary (plus an optional JSON for ``compare.py``) when the run ends."""
    parser = argparse.ArgumentParser(
        description="FLAKE: decentralised P2P federated learning with "
                    "knowledge distillation against a locally-aggregated teacher."
    )
    parser.add_argument(
        "--model", type=int, default=2, choices=list(MODEL_NAME_MAP.keys()),
        help="1=PaperCNN, 2=SimpleCNN (default, matches layer_sharing.py's "
             "SimpleCNN baseline), 3=SimpleCNN10, 4=VGG11-BN, 5=VGG13-BN, "
             "6=VGG16-BN, 7=ResNet-20-CIFAR. Deeper models (4-7) are much "
             "slower per round and benefit most from the KD teacher.",
    )
    global TOTAL_ROUNDS
    parser.add_argument("--rounds", type=int, default=TOTAL_ROUNDS,
                        help="Total federated rounds T (default from FL_ROUNDS env or 100)")
    args = parser.parse_args()
    TOTAL_ROUNDS = int(args.rounds)

    model_name = MODEL_NAME_MAP[args.model]
    log_filename = (
        f"flake_log_{model_name.lower()}_N{NUM_CLIENTS}_T{TOTAL_ROUNDS}_M{NUM_MACHINES}.txt"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(_formatter)
    logger.addHandler(file_handler)

    print(
        f"Starting FLAKE | model={model_name} | N={NUM_CLIENTS} M={NUM_MACHINES} "
        f"T={TOTAL_ROUNDS} | alpha={DIRICHLET_ALPHA} batch={BATCH_SIZE} "
        f"epochs/round={EPOCHS_PER_ROUND}"
    )
    print(f"  current_machine_ip = {CURRENT_MACHINE_IP}")
    print(f"  client_ips         = {ips}")

    local_client_ids = [i for i in range(NUM_CLIENTS) if ips[i] == str(CURRENT_MACHINE_IP)]
    if not local_client_ids:
        print("No clients hosted on this machine; exiting.")
        return
    print(f"Hosting {len(local_client_ids)} local client(s): {local_client_ids}")

    # Build one canonical initial network under a deterministic seed and hand
    # its state to every local client, so round-0's FedAvg is meaningful.
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    _canon = build_model(args.model).to(DEVICE)
    initial_state_np = state_dict_to_numpy(_canon)
    del _canon

    warmup_barrier = threading.Barrier(len(local_client_ids))

    results_store = {}
    results_lock = threading.Lock()
    threads = []
    t_start = time.time()
    for cid in local_client_ids:
        th = threading.Thread(
            target=client_logic,
            args=(cid, args.model, results_store, results_lock,
                  initial_state_np, warmup_barrier),
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    total_time = time.time() - t_start

    # --- Summary ---
    print("\nFLAKE run complete")
    print(f"  Model            : {model_name}")
    print(f"  N / T            : {NUM_CLIENTS} / {TOTAL_ROUNDS}")
    print(f"  Total wall time  : {total_time:.2f}s")
    if local_client_ids:
        print("\nPer-client accuracy summary")
        for cid in sorted(local_client_ids):
            r = results_store.get(cid, {})
            print(
                f"  Client {cid}: final_acc={r.get('final_acc', 0.0):.2f}%, "
                f"best_acc={r.get('best_acc', 0.0):.2f}% (round {r.get('best_round', -1)}), "
                f"train_s={r.get('training_s', 0.0):.1f}, comm_s={r.get('comm_s', 0.0):.1f}"
            )
        avg_final = (sum(float(results_store.get(c, {}).get("final_acc", 0.0))
                         for c in local_client_ids) / max(1, len(local_client_ids)))
        avg_best = (sum(float(results_store.get(c, {}).get("best_acc", 0.0))
                        for c in local_client_ids) / max(1, len(local_client_ids)))
        print(f"\nAverage final accuracy (local clients): {avg_final:.2f}%")
        print(f"Average best  accuracy (local clients): {avg_best:.2f}%")

    # --- Optional JSON results for compare.py ---
    results_path = os.environ.get("FL_RESULTS_JSON")
    if results_path:
        import json
        per_client = {}
        for cid in local_client_ids:
            r = results_store.get(cid, {})
            per_client[str(cid)] = {
                "final_acc": float(r.get("final_acc", 0.0)),
                "final_f1": float(r.get("final_f1", 0.0)),
                "best_acc": float(r.get("best_acc", 0.0)),
                "best_round": int(r.get("best_round", -1)),
                "last_round": int(r.get("last_round", -1)),
                "training_s": float(r.get("training_s", 0.0)),
                "comm_s": float(r.get("comm_s", 0.0)),
                "total_s": float(r.get("total_s", 0.0)),
            }
        avg_final = (sum(p["final_acc"] for p in per_client.values())
                     / max(1, len(per_client)))
        avg_best = (sum(p["best_acc"] for p in per_client.values())
                    / max(1, len(per_client)))
        summary = {
            "framework": "flake",
            "model": model_name,
            "n_clients": NUM_CLIENTS,
            "n_machines": NUM_MACHINES,
            "rounds": TOTAL_ROUNDS,
            "batch_size": BATCH_SIZE,
            "epochs_per_round": EPOCHS_PER_ROUND,
            "dirichlet_alpha": DIRICHLET_ALPHA,
            "total_time_s": float(total_time),
            "avg_final_acc": float(avg_final),
            "avg_best_acc": float(avg_best),
            "per_client": per_client,
        }
        try:
            with open(results_path, "w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"\nWrote results JSON to {results_path}")
        except Exception as e:
            print(f"Failed to write results JSON to {results_path}: {e}")


if __name__ == "__main__":
    main()
