"""Layer Sharing -- decentralised peer-to-peer federated learning on CIFAR-10.

Each round every client

  1. trains its local model for ``E`` SGD epochs on its own CIFAR-10 shard,
  2. randomly assigns every logical layer of its network to itself or to one
     of its peers,
  3. pulls only the assigned layers from those peers,
  4. stacks the pulled tensors into a full state-dict, and
  5. averages that stacked state with its own locally-trained state,
     parameter-by-parameter, before moving on to the next round.

There is no central server and no knowledge distillation -- the only
information that leaves a client is the parameters asked for by peers.  The
early-stopping check watches for ``COUNT_THRESHOLD`` consecutive rounds
with negligible weight change and ends the run cleanly.

Input file (``inputf.txt`` by default; override with ``LAYER_SHARING_INPUT``)::

    N M
    CURRENT_MACHINE_IP
    client_ip_1,client_ip_2,...,client_ip_N
"""

import argparse
import logging
import os
import pickle
import socket
import struct
import sys
import threading
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

np.random.seed(42)
torch.manual_seed(42)

# --- Per-client timing (training vs communication) ----------------------------
_timing_lock = threading.Lock()
client_timing = defaultdict(lambda: {
    "training_s": 0.0,
    "send_s": 0.0,
    "recv_s": 0.0,
    "comm_phase_s": 0.0,
})


def _add_timing(client_id, key, delta_s):
    """Thread-safe accumulator for per-client timing counters."""
    if client_id is None:
        return
    if delta_s is None or delta_s <= 0:
        return
    with _timing_lock:
        client_timing[client_id][key] += float(delta_s)


# --- Constants (overridable via FL_* environment variables) -------------------
# The FL_* names are shared with flake.py so compare.py can align both runs.
BATCH_SIZE = int(os.environ.get("FL_BATCH_SIZE", "32"))
EPOCHS_PER_ROUND = int(os.environ.get("FL_EPOCHS_PER_ROUND", "1"))
DIRICHLET_ALPHA = float(os.environ.get("FL_DIRICHLET_ALPHA", "0.1"))
THRESHOLD = 0.6
FIXED_DATA_PER_CLIENT = 5000
DEVICE = torch.device("cpu")
TIMEOUT = int(os.environ.get("FED_TIMEOUT", "25"))        # per-round pull deadline
CONNECT_TIMEOUT = float(os.environ.get("FED_CONNECT_TIMEOUT", "60"))
TCP_RETRIES = int(os.environ.get("FED_TCP_RETRIES", "1"))
SERVER_BACKLOG = int(os.environ.get("FED_SERVER_BACKLOG", "128"))
R_PRIME = int(os.environ.get("FL_ROUNDS", "100"))           # maximum rounds
COUNT_THRESHOLD = 5                                         # stable rounds needed
# When FL_DISABLE_EARLY_STOP=1 the convergence-based termination check is
# pushed past R_PRIME so the run uses exactly R_PRIME rounds (matches flake.py's
# fixed-T schedule for a fair comparison in compare.py).
MINIMUM_ROUNDS = (
    R_PRIME + 1 if os.environ.get("FL_DISABLE_EARLY_STOP", "0") == "1" else 40
)


# =============================================================================
# Networking utilities (length-prefixed pickle messages over plain TCP)
# =============================================================================

def send_message(conn, message):
    """Pickle ``message`` and send it with a 4-byte big-endian length prefix."""
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    message_length = struct.pack('!I', len(data))
    conn.sendall(message_length + data)


def _recv_exact(conn, nbytes: int):
    """Read exactly ``nbytes`` from ``conn`` (handles partial TCP reads)."""
    data = b''
    while len(data) < nbytes:
        chunk = conn.recv(nbytes - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def receive_message(conn):
    """Inverse of :func:`send_message`: unpack the length prefix then the body."""
    hdr = _recv_exact(conn, 4)
    if not hdr:
        return None
    message_length = struct.unpack('!I', hdr)[0]
    data = _recv_exact(conn, message_length)
    if data is None:
        return None
    return pickle.loads(data)


# =============================================================================
# Input parsing
# =============================================================================

def parse_input_file():
    """Read ``N M``, the current machine's IP, and the N client IPs.

    The input path defaults to ``inputf.txt``; set ``LAYER_SHARING_INPUT`` to
    use a different file (e.g. the localhost layout in compare.py).
    """
    path = os.environ.get("LAYER_SHARING_INPUT", "inputf.txt")
    try:
        with open(path, "r") as file:
            lines = [ln.strip() for ln in file.read().splitlines() if ln.strip()]
            if len(lines) < 3:
                raise ValueError("Input file does not contain enough lines.")

            num_clients, num_machines = map(int, lines[0].split())
            current_machine_ip = lines[1].strip()
            all_ips = [ip.strip() for ip in lines[2].split(",")]
            if len(all_ips) != num_clients:
                raise ValueError(
                    f"Expected {num_clients} client IPs on line 3, got {len(all_ips)}."
                )

        return num_clients, num_machines, current_machine_ip, all_ips
    except FileNotFoundError:
        print("The input file was not found.")
    except ValueError as ve:
        print(f"Error parsing input file: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None, None, None, None


# --- Module-level setup ---

NUM_CLIENTS, NUM_MACHINES, CURRENT_MACHINE_IP, ips = parse_input_file()

if NUM_CLIENTS is None:
    print("Failed to parse the input file. Exiting.")
    exit(1)

# Logger: console handler added now; file handler added in main() once the
# model name is known.
logger = logging.getLogger('federated_learning')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class LoggerWriter:
    """Tiny stdout replacement that forwards print() calls through the logger."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


sys.stdout = LoggerWriter(logger, logging.INFO)

retries_list = [1] * NUM_CLIENTS
adj = [[j for j in range(NUM_CLIENTS) if j != i] for i in range(NUM_CLIENTS)]
terminate_messages = [0] * NUM_CLIENTS
model_messages = [0] * NUM_CLIENTS

# CIFAR-10 dataset -- same seeded Dirichlet split as flake.py at the same alpha.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

indices = np.random.permutation(len(train_dataset))


def create_dirichlet_non_iid_splits_fixed(dataset, num_clients, alpha=0.1, fixed_data_per_client=5000):
    """Split ``dataset`` into ``num_clients`` non-IID shards via a class-aware
    Dirichlet(alpha) draw, then truncate / up-sample each shard to exactly
    ``fixed_data_per_client`` examples. Lower alpha -> more label skew.
    """
    num_classes = 10
    class_indices = {i: np.where(np.array(dataset.targets) == i)[0] for i in range(num_classes)}
    client_indices = {i: [] for i in range(num_clients)}

    for c, idxs in class_indices.items():
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(idxs)).astype(int)
        start_idx = 0
        for i, count in enumerate(proportions):
            client_indices[i].extend(idxs[start_idx:start_idx + count])
            start_idx += count

    final_client_indices = {}
    for client_id, idxs in client_indices.items():
        np.random.shuffle(idxs)
        if len(idxs) > fixed_data_per_client:
            final_client_indices[client_id] = idxs[:fixed_data_per_client]
        else:
            final_client_indices[client_id] = np.random.choice(idxs, fixed_data_per_client, replace=True).tolist()

    client_data = [torch.utils.data.Subset(dataset, final_client_indices[i]) for i in range(num_clients)]
    return client_data


client_data = create_dirichlet_non_iid_splits_fixed(
    train_dataset, NUM_CLIENTS, alpha=DIRICHLET_ALPHA, fixed_data_per_client=FIXED_DATA_PER_CLIENT
)

msg_lck = threading.Lock()
latest_models_lock = threading.Lock()


# =============================================================================
# Model definitions (mirror flake.py for apples-to-apples comparisons)
# =============================================================================

class SimpleCNN(nn.Module):
    """Small 3x3 CNN baseline (two conv layers, ~0.4M params)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


class SimpleCNN10(nn.Module):
    """10-layer CNN (4 conv blocks + 2 FC), ~4.7M params."""

    def __init__(self, num_classes=10):
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
            nn.Linear(512 * 2 * 2, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
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
    """CIFAR-sized VGG (1x1 spatial output after 5 max-pools) with BN + dropout."""

    def __init__(self, features, num_classes=10):
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
    """Constructor for VGG-11 with batch norm."""
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def VGG13BN():
    """Constructor for VGG-13 with batch norm."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


def VGG16BN():
    """Constructor for VGG-16 with batch norm."""
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    return VGG(_make_vgg_layers(cfg))


class _CifarBasicBlock(nn.Module):
    """Two 3x3 convs + BN + ReLU with a projection shortcut when dims change."""

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return torch.relu(out)


class _CifarResNet(nn.Module):
    """CIFAR ResNet (He et al. 2015). Depth = 6n + 2; n=3 gives ResNet-20."""

    def __init__(self, n=3, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, n, stride=1)
        self.layer2 = self._make_layer(32, n, stride=2)
        self.layer3 = self._make_layer(64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, planes, n, stride):
        """Stack ``n`` basic blocks; the first may down-sample via ``stride``."""
        layers = [_CifarBasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(n - 1):
            layers.append(_CifarBasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.fc(out)


def ResNet20():
    """ResNet-20 for CIFAR (20 layers, ~0.27M params)."""
    return _CifarResNet(n=3)


def build_model(choice: int) -> nn.Module:
    """Instantiate the model selected by ``--model`` (see MODEL_NAME_MAP)."""
    if choice == 1:
        return SimpleCNN()
    if choice == 2:
        return SimpleCNN10()
    if choice == 3:
        return VGG11BN()
    if choice == 4:
        return VGG13BN()
    if choice == 5:
        return VGG16BN()
    if choice == 6:
        return ResNet20()
    raise ValueError("Model choice must be 1..6")


MODEL_NAME_MAP = {
    1: "SimpleCNN",
    2: "SimpleCNN10",
    3: "VGG11-BN",
    4: "VGG13-BN",
    5: "VGG16-BN",
    6: "ResNet-20-CIFAR",
}


# =============================================================================
# State-dict / parameter helpers
# =============================================================================

def _state_dict_to_numpy(model: nn.Module):
    """Snapshot a model's state_dict as pure numpy (safe to pickle and send)."""
    sd = model.state_dict()
    return {k: v.detach().cpu().numpy() for k, v in sd.items()}


def _numpy_to_state_dict_torch(state_np):
    """Convert a numpy state_dict back to torch tensors on ``DEVICE``."""
    return {k: torch.tensor(v).to(DEVICE) for k, v in state_np.items()}


def _logical_layer_key(param_name: str) -> str:
    """Return the parent module name for a parameter key.

    e.g. ``conv1.weight`` and ``conv1.bias`` both collapse to ``conv1`` so
    they get assigned to the same peer in the random layer-stacking step.
    """
    if '.' not in param_name:
        return param_name
    return param_name.rsplit('.', 1)[0]


def _group_params_by_logical_layer(state_np):
    """Group the keys of a state_dict by their logical layer."""
    groups = defaultdict(list)
    for name in state_np.keys():
        groups[_logical_layer_key(name)].append(name)
    return dict(groups)


def _state_dict_to_list_sorted(state_np):
    """Return the state_dict's tensors in a stable (sorted-key) order."""
    return [state_np[k] for k in sorted(state_np.keys())]


def models_are_similar_list(weights1_list, weights2_list, threshold):
    """Convergence check: are two weight lists everywhere within ``threshold`` in L2?"""
    for w1, w2 in zip(weights1_list, weights2_list):
        if np.linalg.norm(w1 - w2) > threshold:
            return False
    return True


def compute_accuracy(model, data_loader):
    """Return top-1 accuracy (%) of ``model`` on ``data_loader``."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100 * correct / total


# =============================================================================
# Networking
# =============================================================================

def tcp_client(id, target_id, target_ip, message):
    """Fire-and-forget send (terminate / weights broadcast) with retries."""
    global retries_list
    retries = TCP_RETRIES
    while retries > 0:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(CONNECT_TIMEOUT)
        try:
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
            client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        except Exception:
            pass
        try:
            client.connect((target_ip, 8650 + target_id))
            send_message(client, message)
            return True
        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError, socket.timeout):
            retries -= 1
            retries_list[target_id] -= 1
            time.sleep(1)
        finally:
            try:
                client.close()
            except Exception:
                pass
    return False


def tcp_client_request_layers(requester_id, target_id, target_ip, param_names, current_round, deadline_ts=None):
    """Pull a specific set of named parameters from peer ``target_id``.

    Returns the ``{param_name: numpy_array}`` dict sent back by the peer, or
    ``None`` on timeout / refused / round mismatch.
    """
    global retries_list
    if not param_names:
        return None

    retries = TCP_RETRIES
    while retries > 0:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            if deadline_ts is not None:
                timeout_s = max(0.1, float(deadline_ts - time.time()))
                client.settimeout(timeout_s)
            else:
                client.settimeout(CONNECT_TIMEOUT)
            try:
                client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
                client.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
            except Exception:
                pass

            _t_send0 = time.perf_counter()
            client.connect((target_ip, 8650 + target_id))
            with msg_lck:
                model_messages[requester_id] += 1

            send_message(
                client,
                {
                    'type': 'layer_request',
                    'requester_id': requester_id,
                    'id': requester_id,
                    'round': current_round,
                    'params': list(param_names),
                },
            )
            _add_timing(requester_id, "send_s", time.perf_counter() - _t_send0)

            _t_recv0 = time.perf_counter()
            resp = receive_message(client)
            _add_timing(requester_id, "recv_s", time.perf_counter() - _t_recv0)

            if resp is None:
                return None
            if resp.get('type') != 'layer_response':
                return None
            if resp.get('round') != current_round:
                return None
            return resp.get('params', None)

        except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError, socket.timeout):
            retries -= 1
            retries_list[target_id] -= 1
            time.sleep(0.05)
        finally:
            try:
                client.close()
            except Exception:
                pass
        if retries == 0:
            break
    return None


def broadcast_weights(id, weights_state_np, current_round, terminate, ips, latest_models):
    """Push this client's current state_dict to every peer (used at termination)."""
    global model_messages
    message = {
        'type': 'weights',
        'weights': weights_state_np,
        'round': current_round,
        'terminate': terminate,
        'id': id,
    }
    for pid in adj[id]:
        with msg_lck:
            model_messages[id] += 1
        tcp_client(id, pid, ips[pid], message)
    latest_models[id] = weights_state_np


def broadcast_terminate(id, ips):
    """Tell every peer that this client is done (stops their round loop)."""
    global terminate_messages
    message = {'type': 'terminate'}
    for pid in adj[id]:
        terminate_messages[id] += 1
        tcp_client(id, pid, ips[pid], message)


def tcp_server(id, terminate_flags, local_ip, latest_models, stop_event):
    """Serve incoming peer requests until ``stop_event`` fires.

    Two message types are understood:
      * ``terminate``     -- set ``stop_event`` so the client's loop exits;
      * ``layer_request`` -- reply with the subset of the client's latest
        parameters named in the request.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 8650 + id))
    server.listen(SERVER_BACKLOG)
    server.settimeout(1.0)

    def handle_conn(conn):
        try:
            try:
                conn.settimeout(CONNECT_TIMEOUT)
            except Exception:
                pass
            msg = receive_message(conn)
            if not msg:
                return

            if msg.get('type') == 'terminate':
                terminate_flags.append(1)
                stop_event.set()
                return

            if msg.get('type') == 'layer_request':
                requested_params = msg.get('params', [])
                round_id = msg.get('round', None)

                with latest_models_lock:
                    local_snapshot = latest_models.get(id, None)
                    if local_snapshot is None or not isinstance(local_snapshot, dict):
                        local_snapshot = {}

                payload = {
                    pname: local_snapshot[pname]
                    for pname in requested_params
                    if pname in local_snapshot
                }

                _t_send0 = time.perf_counter()
                send_message(
                    conn,
                    {
                        'type': 'layer_response',
                        'provider_id': id,
                        'id': id,
                        'round': round_id,
                        'params': payload,
                    },
                )
                _add_timing(id, "send_s", time.perf_counter() - _t_send0)
        finally:
            try:
                conn.close()
            except Exception:
                pass

    try:
        while not stop_event.is_set():
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                continue
            _t_recv0 = time.perf_counter()
            _add_timing(id, "recv_s", time.perf_counter() - _t_recv0)
            threading.Thread(target=handle_conn, args=(conn,), daemon=True).start()
    finally:
        try:
            server.close()
        except Exception:
            pass


# =============================================================================
# Random layer-stacking helpers
# =============================================================================

def _random_assign_layers_to_participants(local_id, participant_ids, layer_groups):
    """Randomly assign every logical layer to exactly one participant (possibly self)."""
    assignment = {}
    for layer_key in layer_groups.keys():
        assignment[layer_key] = int(np.random.choice(participant_ids))
    return assignment


def _random_stack_and_average(local_id, local_state_np, peer_state_by_id, layer_groups):
    """Stack one participant's version of each logical layer, then average with self.

    Kept as a convenience helper; the actual round loop uses the pull-based
    variant directly in :func:`client_logic`.
    """
    participants = sorted(set(peer_state_by_id.keys()) | {local_id})
    models_by_id = dict(peer_state_by_id)
    models_by_id[local_id] = local_state_np

    stacked_state = dict(local_state_np)
    for layer_key, param_names in layer_groups.items():
        chosen = int(np.random.choice(participants))
        chosen_state = models_by_id.get(chosen, local_state_np)
        for pname in param_names:
            if pname in chosen_state:
                stacked_state[pname] = chosen_state[pname]

    return {k: (local_state_np[k] + stacked_state[k]) / 2.0 for k in local_state_np.keys()}


# =============================================================================
# Per-client round loop
# =============================================================================

def client_logic(id, local_ip, ips, model_choice, timing_store):
    """Run the full Layer Sharing training loop for a single client.

    Boots a TCP listener thread, then for each round: train locally, make the
    resulting parameters pullable, randomly assign each logical layer to a
    participant, pull the assigned layers from the chosen peers, average the
    stacked state with the locally-trained state, and evaluate.  Terminates
    early once the weights have been stable for ``COUNT_THRESHOLD`` rounds
    (unless ``FL_DISABLE_EARLY_STOP=1``).
    """
    model = build_model(model_choice).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader = torch.utils.data.DataLoader(client_data[id], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    previous_weights_list = None
    current_round = 0
    terminate_flags = []
    counter = 0
    latest_models = defaultdict(dict)
    best_accuracy = 0.0
    best_round = -1
    final_accuracy = 0.0

    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=tcp_server,
        args=(id, terminate_flags, local_ip, latest_models, stop_event),
    )
    server_thread.start()
    time.sleep(2)

    while current_round < R_PRIME:
        # ---- Local training ----
        _t_train0 = time.perf_counter()
        model.train()
        for _epoch in range(EPOCHS_PER_ROUND):
            for data, target in train_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(data), target)
                loss.backward()
                optimizer.step()
        _add_timing(id, "training_s", time.perf_counter() - _t_train0)

        local_state_np = _state_dict_to_numpy(model)
        layer_groups = _group_params_by_logical_layer(local_state_np)

        # Publish the locally-trained parameters for peers to pull from.
        with latest_models_lock:
            latest_models[id] = local_state_np

        # ---- Termination from peer ----
        if terminate_flags:
            print(f"Client {id} received termination flag at round {current_round}")
            broadcast_terminate(id, ips)
            break

        # ---- Random layer assignment (pull-based) ----
        participants = [id] + adj[id]
        assignment = _random_assign_layers_to_participants(id, participants, layer_groups)

        params_needed_by_peer = defaultdict(list)
        for layer_key, chosen_peer in assignment.items():
            if chosen_peer == id:
                continue
            params_needed_by_peer[chosen_peer].extend(layer_groups[layer_key])

        deadline_ts = time.time() + TIMEOUT
        pulled_params_by_peer = {}

        _t_comm0 = time.perf_counter()
        for peer_id, param_names in params_needed_by_peer.items():
            if time.time() >= deadline_ts:
                break
            resp_params = tcp_client_request_layers(
                requester_id=id,
                target_id=peer_id,
                target_ip=ips[peer_id],
                param_names=param_names,
                current_round=current_round,
                deadline_ts=deadline_ts,
            )
            if resp_params is not None:
                pulled_params_by_peer[peer_id] = resp_params
        _add_timing(id, "comm_phase_s", time.perf_counter() - _t_comm0)

        # ---- Build stacked state from pulled layers ----
        stacked_state = dict(local_state_np)
        for layer_key, chosen_peer in assignment.items():
            if chosen_peer == id:
                continue
            resp = pulled_params_by_peer.get(chosen_peer, {})
            for pname in layer_groups[layer_key]:
                if pname in resp:
                    stacked_state[pname] = resp[pname]

        # Average the stacked (peer-stitched) state with the local state
        # parameter-by-parameter.
        new_state_np = {k: (local_state_np[k] + stacked_state[k]) / 2.0 for k in local_state_np.keys()}
        model.load_state_dict(_numpy_to_state_dict_torch(new_state_np), strict=True)

        accuracy = compute_accuracy(model, test_loader)
        final_accuracy = float(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = float(accuracy)
            best_round = int(current_round)
        print(f"Client {id} - Round {current_round}: Accuracy: {accuracy:.2f}%")

        new_weights_list = _state_dict_to_list_sorted(new_state_np)

        # ---- Early-stopping check (convergence by weight stability) ----
        if current_round >= MINIMUM_ROUNDS:
            if previous_weights_list is not None and models_are_similar_list(
                new_weights_list, previous_weights_list, THRESHOLD
            ):
                counter += 1
            else:
                counter = 0

            if counter >= COUNT_THRESHOLD:
                print(
                    f"Client {id} met termination criteria at round {current_round}: "
                    f"stable weights for {COUNT_THRESHOLD} rounds"
                )
                broadcast_weights(
                    id, local_state_np, current_round, terminate=1,
                    ips=ips, latest_models=latest_models,
                )
                break

        previous_weights_list = new_weights_list
        current_round += 1

    if current_round == R_PRIME:
        print(f"Client {id} reached maximum {R_PRIME} rounds and is terminating")
        broadcast_terminate(id, ips)

    print(f"Client {id} finished.")

    with _timing_lock:
        t = dict(client_timing.get(id, {}))
    train_s = float(t.get("training_s", 0.0))
    send_s = float(t.get("send_s", 0.0))
    recv_s = float(t.get("recv_s", 0.0))
    comm_phase_s = float(t.get("comm_phase_s", 0.0))
    comm_io_s = send_s + recv_s
    total_s = train_s + comm_io_s + comm_phase_s
    print(
        f"Client {id} timing (s): train={train_s:.2f}, "
        f"comm_io={comm_io_s:.2f} [send {send_s:.2f}, recv {recv_s:.2f}], "
        f"comm_phase={comm_phase_s:.2f}, "
        f"comm_total={comm_io_s + comm_phase_s:.2f}, "
        f"total={total_s:.2f}"
    )

    timing_store[id] = {
        "training_s": train_s,
        "comm_io_s": comm_io_s,
        "send_s": send_s,
        "recv_s": recv_s,
        "comm_phase_s": comm_phase_s,
        "total_s": total_s,
        "final_acc": final_accuracy,
        "best_acc": best_accuracy,
        "best_round": best_round,
        "last_round": int(current_round),
    }

    broadcast_terminate(id, ips)
    stop_event.set()
    server_thread.join()


# =============================================================================
# Entry point
# =============================================================================

def main():
    """Parse CLI args, spawn one thread per locally-hosted client, and write a
    summary (plus an optional JSON for ``compare.py``) when the run ends."""
    parser = argparse.ArgumentParser(
        description="Federated Learning -- Random Layer Stacking (selectable model)"
    )
    parser.add_argument(
        "--model",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help=(
            "1 = SimpleCNN (small/original), "
            "2 = SimpleCNN10 (deep 10-layer CNN), "
            "3 = VGG11-BN, "
            "4 = VGG13-BN, "
            "5 = VGG16-BN, "
            "6 = ResNet-20-CIFAR (~0.27M params)"
        ),
    )
    args = parser.parse_args()
    model_name = MODEL_NAME_MAP[args.model]

    # Attach file handler now that the model name is known.
    log_filename = (
        f"layer_sharing_log_{model_name.lower().replace('-', '_')}"
        f"_N{NUM_CLIENTS}_M{NUM_MACHINES}.txt"
    )
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    global model_messages, terminate_messages
    start_time = time.time()
    print(f"Starting Federated Learning -- Random Layer Stacking | Model: {model_name}")

    threads = []
    timing_store = {}
    for i in range(NUM_CLIENTS):
        if ips[i] == str(CURRENT_MACHINE_IP):
            t = threading.Thread(
                target=client_logic,
                args=(i, CURRENT_MACHINE_IP, ips, args.model, timing_store),
            )
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    total_time = end_time - start_time

    total_model_messages = sum(model_messages)
    total_termination_messages = sum(terminate_messages)

    print("\nFederated Learning Completed")
    print(f"Current Machine IP : {CURRENT_MACHINE_IP}")
    print(f"Number of Clients  : {NUM_CLIENTS}")
    print(f"Model              : {model_name}")
    print(f"Total layer requests sent : {total_model_messages}")
    print(f"Total termination messages: {total_termination_messages}")
    print(f"Total time taken   : {total_time:.2f} seconds")

    local_client_ids = sorted(
        i for i in range(NUM_CLIENTS) if ips[i] == str(CURRENT_MACHINE_IP)
    )
    if local_client_ids:
        print("\nPer-client timing summary (seconds)")
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            train_s = float(t.get("training_s", 0.0))
            comm_io_s = float(t.get("comm_io_s", 0.0))
            send_s = float(t.get("send_s", 0.0))
            recv_s = float(t.get("recv_s", 0.0))
            comm_phase_s = float(t.get("comm_phase_s", 0.0))
            total_s = float(t.get("total_s", 0.0))
            print(
                f"  Client {cid}: train={train_s:.2f}, "
                f"comm_io={comm_io_s:.2f} [send {send_s:.2f}, recv {recv_s:.2f}], "
                f"comm_phase={comm_phase_s:.2f}, "
                f"comm_total={comm_io_s + comm_phase_s:.2f}, "
                f"total={total_s:.2f}"
            )

        print("\nPer-client accuracy summary")
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            final_acc = float(t.get("final_acc", 0.0))
            best_acc = float(t.get("best_acc", 0.0))
            best_rnd = int(t.get("best_round", -1))
            last_rnd = int(t.get("last_round", -1))
            print(
                f"  Client {cid}: final_acc={final_acc:.2f}%, "
                f"best_acc={best_acc:.2f}% (round {best_rnd}), "
                f"last_round={last_rnd}"
            )

        avg_final_acc = (
            sum(float(timing_store.get(cid, {}).get("final_acc", 0.0)) for cid in local_client_ids)
            / max(1, len(local_client_ids))
        )
        print(f"\nAverage final accuracy (local clients): {avg_final_acc:.2f}%")

        agg_train = sum(float(timing_store.get(cid, {}).get("training_s", 0.0)) for cid in local_client_ids)
        agg_comm_io = sum(float(timing_store.get(cid, {}).get("comm_io_s", 0.0)) for cid in local_client_ids)
        agg_comm_phase = sum(float(timing_store.get(cid, {}).get("comm_phase_s", 0.0)) for cid in local_client_ids)
        agg_total = sum(float(timing_store.get(cid, {}).get("total_s", 0.0)) for cid in local_client_ids)
        n = max(1, len(local_client_ids))
        print("\nAggregate timing (local clients)")
        print(
            f"  Sum  : train={agg_train:.2f}, comm_io={agg_comm_io:.2f}, "
            f"comm_phase={agg_comm_phase:.2f}, total={agg_total:.2f}"
        )
        print(
            f"  Avg  : train={agg_train/n:.2f}, comm_io={agg_comm_io/n:.2f}, "
            f"comm_phase={agg_comm_phase/n:.2f}, total={agg_total/n:.2f}"
        )

    # ---- Optional: dump a JSON results file for comparison harnesses. ----
    results_path = os.environ.get("FL_RESULTS_JSON")
    if results_path:
        import json
        per_client = {}
        for cid in local_client_ids:
            t = timing_store.get(cid, {})
            per_client[str(cid)] = {
                "final_acc": float(t.get("final_acc", 0.0)),
                "best_acc": float(t.get("best_acc", 0.0)),
                "best_round": int(t.get("best_round", -1)),
                "last_round": int(t.get("last_round", -1)),
                "training_s": float(t.get("training_s", 0.0)),
                "comm_io_s": float(t.get("comm_io_s", 0.0)),
                "comm_phase_s": float(t.get("comm_phase_s", 0.0)),
                "total_s": float(t.get("total_s", 0.0)),
            }
        avg_final = (
            sum(pc["final_acc"] for pc in per_client.values()) / max(1, len(per_client))
        )
        avg_best = (
            sum(pc["best_acc"] for pc in per_client.values()) / max(1, len(per_client))
        )
        summary = {
            "framework": "layer_sharing",
            "model": model_name,
            "n_clients": NUM_CLIENTS,
            "n_machines": NUM_MACHINES,
            "max_rounds": R_PRIME,
            "early_stopping_disabled": os.environ.get("FL_DISABLE_EARLY_STOP", "0") == "1",
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
