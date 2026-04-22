# Federated learning with Asynchronous Non-IID Data & Knowledge Exchange


Mini-project comparing two **serverless, peer-to-peer** federated-learning
protocols on CIFAR-10 under a non-IID Dirichlet split:

- **FLAKE** (`flake.py`) — *Federated Learning with Aggregated Knowledge-Distillation
  Exchange*. Every round each client pulls every peer's weights, FedAvgs them
  into a frozen teacher, and trains its student with
  `L = CE + mu * KL(student || teacher)`.
- **Layer Sharing** (`layer_sharing.py`) — each client trains locally, randomly
  assigns every logical layer to itself or a peer, pulls only those layers, and
  averages the stacked state-dict with its own.

Both run without a central server; all communication is length-prefixed pickle
over plain TCP between clients. A single harness (`compare.py`) runs them
back-to-back under identical hyperparameters and emits a slide-ready log.

---

## Repository layout

```
flake.py                       # FLAKE framework (selectable model, --model 1..7)
layer_sharing.py               # Layer Sharing framework (--model 1..6)
compare.py                     # End-to-end sweep harness (both frameworks)
flake_input.txt                # Multi-machine input for FLAKE
flake_input_localhost.txt      # Single-host localhost input for FLAKE
layer_sharing_input.txt        # Multi-machine input for Layer Sharing
layer_sharing_input_localhost.txt
data/cifar-10-batches-py/      # CIFAR-10 (auto-downloaded on first run)
Documents/                     # Slide decks (FLAKE.pptx, FedAck.pptx)
presentation-content.md        # Long-form write-up of results
presentation_results.log       # Latest harness output
FLonNonIIDviaRLnKnowledgeDistill.pdf  # Reference paper
```

---

## Requirements

- Python 3.10+
- PyTorch, torchvision, NumPy

```bash
pip install torch torchvision numpy
```

CIFAR-10 is fetched into `data/` on first run (the tarball is already present,
so no network access is required).

---

## Input file format

Both frameworks read the same 3-line format. FLAKE uses `flake_input.txt` (or
`$FLAKE_INPUT`); Layer Sharing uses `inputf.txt` (or `$LAYER_SHARING_INPUT`):

```
N M
CURRENT_MACHINE_IP
client_ip_1,client_ip_2,...,client_ip_N
```

- `N` — total number of logical clients across the deployment.
- `M` — number of physical machines.
- `CURRENT_MACHINE_IP` — IP of the host you're launching on (every client whose
  IP in the list matches this one is threaded into this process).
- The third line lists every client's host IP, in order; duplicates are
  expected when stacking multiple clients per machine.

Example `flake_input_localhost.txt` (6 clients on one host):

```
6 1
127.0.0.1
127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1,127.0.0.1
```

Example `flake_input.txt` (6 clients split across 3 machines, launched from
`10.0.0.3`):

```
6 3
10.0.0.3
10.0.0.1,10.0.0.1,10.0.0.2,10.0.0.2,10.0.0.3,10.0.0.3
```

FLAKE listens on `FLAKE_BASE_PORT + client_id` (default `8700`). Layer Sharing
uses its own port range, so both frameworks can coexist on one host.

---

## Running FLAKE

```bash
python flake.py --model 2 --rounds 100
```

`--model` choices (mirrored in `MODEL_NAME_MAP`):

| ID | Model            | Notes                                               |
|----|------------------|-----------------------------------------------------|
| 1  | PaperCNN         | Original paper CNN                                  |
| 2  | SimpleCNN        | Default; matches Layer Sharing's SimpleCNN baseline |
| 3  | SimpleCNN10      | Deeper 10-layer CNN                                 |
| 4  | VGG11-BN         | ~10M params, slow                                   |
| 5  | VGG13-BN         | ~10M params, slow                                   |
| 6  | VGG16-BN         | ~15M params, slowest                                |
| 7  | ResNet-20-CIFAR  | ~0.27M params, strong baseline                      |

Override the default input file:

```bash
FLAKE_INPUT=flake_input.txt python flake.py --model 7 --rounds 60
```

### FLAKE hyperparameters (env vars)

| Var                          | Default | Meaning                                    |
|------------------------------|---------|--------------------------------------------|
| `FL_BATCH_SIZE`              | 32      | Local SGD batch size                       |
| `FL_EPOCHS_PER_ROUND`        | 1       | Local epochs `E` per round                 |
| `FL_ROUNDS`                  | 100     | Total federated rounds `T`                 |
| `FL_DIRICHLET_ALPHA`         | 0.5     | Non-IID split concentration                |
| `FL_FLAKE_TAU`               | 1.0     | KD softmax temperature                     |
| `FL_FLAKE_MU`                | 1.5     | KD KL weight vs. cross-entropy             |
| `FL_FLAKE_WARMUP_EPOCHS`     | 1       | CE-only warm-up epochs before round 0      |
| `FL_FLAKE_KD_WARMUP_ROUNDS`  | 3       | Rounds to ramp `mu` from 0 to `FL_FLAKE_MU`|
| `FLAKE_BASE_PORT`            | 8700    | First TCP port (client `i` → base + i)     |

---

## Running Layer Sharing

Layer Sharing defaults to `inputf.txt`. Either rename or point at one of the
supplied input files:

```bash
LAYER_SHARING_INPUT=layer_sharing_input_localhost.txt \
  python layer_sharing.py --model 1
```

`--model` choices:

| ID | Model           |
|----|-----------------|
| 1  | SimpleCNN       |
| 2  | SimpleCNN10     |
| 3  | VGG11-BN        |
| 4  | VGG13-BN        |
| 5  | VGG16-BN        |
| 6  | ResNet-20-CIFAR |

Shares the same `FL_*` env vars as FLAKE (batch size, epochs per round,
rounds, alpha). Set `FL_DISABLE_EARLY_STOP=1` to force exactly `FL_ROUNDS`
rounds for an apples-to-apples comparison; otherwise Layer Sharing exits
after `COUNT_THRESHOLD` consecutive rounds with negligible weight change.

---

## End-to-end comparison harness

`compare.py` runs both frameworks on the same Dirichlet split under identical
hyperparameters and writes a presentation-ready log (default
`presentation_results.log`) plus a resumable state file.

```bash
python compare.py                            # default sweep
python compare.py --rounds 100               # longer runs
python compare.py --ls-models 1,2,3,4,5,6 \  # full Layer Sharing zoo
                  --fk-models 1,2,3,4,5,6,7  # full FLAKE zoo
python compare.py --fresh                    # ignore resume state
python compare.py --skip-flake               # layer_sharing only
python compare.py --log my_results.log       # custom log path
```

Defaults (`--ls-models 1,2,6 --fk-models 2,3,7`) pair up as equivalent
architectures: SimpleCNN ↔ SimpleCNN, SimpleCNN10 ↔ SimpleCNN10,
ResNet-20 ↔ ResNet-20.

The harness maintains `<log>.state.json` next to the log and rewrites it after
every completed run — re-running with the same flags resumes where it left
off. To keep a long sweep alive on macOS with the lid closed:

```bash
caffeinate -s python compare.py --rounds 100
```

---

## Reproducibility

Both frameworks seed Python/NumPy/PyTorch with `SEED=42`, build the **same**
initial network under that seed, and hand identical initial weights to every
local client. Combined with a fixed Dirichlet alpha, this makes round 0's
aggregation well-defined and makes two runs of the harness reproduce to within
TCP-scheduling jitter.

---

## Output artefacts

- `flake_log_<model>_N<N>_T<T>_M<M>.txt` — FLAKE per-run log.
- `layer_sharing_log_<model>_N<N>_M<M>.txt` — Layer Sharing per-run log.
- `compare_<framework>_m<id>.log` — harness-captured stdout for each child run.
- `<framework>_summary_m<id>.json` — machine-readable per-run summary consumed
  by `compare.py`.
- `presentation_results.log` (+ `.state.json`) — rolling slide-ready summary.
