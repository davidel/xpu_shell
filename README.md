# XPU Shell (`xpush`)

A unified command-line interface to provision, execute, and supervise workloads on AI accelerators across multiple cloud providers (**RunPod**, **Vast.ai**, **Spheron**, etc.) with persistent network volumes, customized container environments, and automatic SSH setup.

## Features

- **Multi-Cloud Accelerator Support**: Manage instances across **RunPod**, **Vast.ai**, and **Spheron** using a single unified CLI, with automatic provider detection from instance IDs or state.
- **Fast CLI Ergonomics (`xpush`)**: Provided with short `xpush` and full `xpu-shell` command aliases (plus backward-compatible `runpod-shell`).
- **Docker-like Subcommands**: Simple interface to `create`, `list`, `info`, `stop`, `terminate`, `templates`, `gpus`, `run`, `exec`, and `cp`.
- **Rich Instance Inspection (`info`)**: Inspect comprehensive specifications including GPU model, VRAM size, allocated vCPUs, system RAM, disk volumes, live utilization metrics (CPU %, RAM %, GPU %, VRAM %), uptime, cost, and direct SSH connection details.
- **Automatic Instance & Job Memory**: Automatically remembers the last created instance ID and provider in `~/.config/xpu_shell/.last_pod_id` and the last executed job ID in `~/.config/xpu_shell/.last_job_id`. All commands (`run`, `exec`, `ps`, `logs`, `kill`, `cp`, `stop`, `terminate`) work seamlessly without having to re-type the instance ID.

- **Seamless File Transfers (`cp`)**: Transfer files and directories to and from remote instances via `scp` with intuitive `[instance_id:]path` and `:/path` syntax, automatic SSH port & key resolution, recursive copying (`-r`), and attribute preservation (`-P`).
- **In-Memory Secret & Environment Injection**: Pass sensitive secrets (e.g. S3/GCS keys, R2 tokens) into remote scripts in-memory via SSH with `--env` / `-e` or `--env-file` without persisting credentials to the remote disk.
- **Graceful Job Termination**: Kills remote jobs by sending `SIGTERM` first, monitoring exit status, and escalating to `SIGKILL` after a configurable timeout (default: 30s).
- **GPU Querying & Marketplace Resolution**: List available GPU types and pricing using `gpus` with regex filtering across RunPod or Vast.ai.
- **Smart SSH Key Auto-Detection**: Searches for standard SSH public keys (`id_rsa.pub`, `id_ed25519.pub`, `id_ecdsa.pub`, `id_dsa.pub`) in your local `~/.ssh/` directory automatically.
- **SSH Isolation & Sandbox Support**: Support for custom SSH config files via `--ssh-config` or `RUNPOD_SSH_CONFIG` / `XPU_SSH_CONFIG` (e.g. `/dev/null`).

---

## Installation

1. **Install the package**:

   From source:
   ```bash
   pip install .
   ```
   Or for editable development mode:
   ```bash
   pip install -e .
   ```

2. **Environment Variables**:
   Set your provider API keys as needed:
   ```bash
   # For RunPod:
   export RUNPOD_API_KEY="your_runpod_api_key"

   # For Vast.ai:
   export VAST_API_KEY="your_vast_api_key"
   # (Vast also automatically detects ~/.vast_api_key)

   # For Spheron:
   export SPHERON_API_KEY="your_spheron_api_key"
   # (Spheron also automatically detects ~/.spheron_api_key)
   ```

   *(Optional)* Configure or isolate the persistence directory (defaults to `~/.config/xpu_shell`):
   ```bash
   export XPU_SHELL_CONFIG_DIR="/path/to/config"
   ```

> **Note:** Once installed, both `xpush` and `xpu-shell` are available in your `PATH`.


---

## Command Reference

### Global Options
Every subcommand supports the following options:
* `--api-key`: API key for the active provider (RunPod, Vast.ai, or Spheron). Defaults to `RUNPOD_API_KEY`, `VAST_API_KEY` / `~/.vast_api_key`, or `SPHERON_API_KEY` / `~/.spheron_api_key`.
* `--provider`: Target provider (`runpod`, `vast`, or `spheron`). Defaults to auto-detecting from the target ID, the last used provider in `~/.config/xpu_shell/.last_provider`, or `runpod`.

---

### 1. `create`
Launches and configures a new instance on RunPod or Vast.ai.

```bash
xpush create [OPTIONS]
```

#### Common Options (All Providers)
| Flag | Default | Description |
|---|---|---|
| `--provider` | `runpod` | Target cloud provider (`runpod` or `vast`). Auto-detected if `--offer-id` is given. |
| `--name` | `persistent-worker` | Name of the instance |
| `--image-name` | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` | Base image for the container (defaults to `pytorch/pytorch:latest` on Vast) |
| `--gpu-type` | `NVIDIA GeForce RTX 4090` | GPU type to provision (supports substring and fuzzy matching, e.g. `4090`) |
| `--gpu-count` | `1` | Number of GPUs to allocate |
| `--volume-size` | `50` | Volume disk size in GB |
| `--ssh-key-path` | *None* | Path to public key (checks default paths if omitted) |
| `--requirements-path` | *None* | Path to `requirements.txt` file |
| `--pip-packages` | *None* | Extra Python packages to install |
| `--apt-packages` | *None* | Extra apt packages to install (default: `screen curl htop ffmpeg git`) |
| `--apt-packages-file` | *None* | Path to a file containing extra apt packages to install |
| `--env` | *None* | Environment variables (`KEY=VALUE`, or RunPod secret references `KEY='{{ RUNPOD_SECRET_<NAME> }}'`) |
| `--env-file` | *None* | Path to a `.env` file containing environment variables |
| `--container-disk-size` | `30` | Container local disk size in GB |
| `--volume-mount-path` | `/workspace` | Path inside container where the volume is mounted |
| `--run-script` | *None* | Path to a local script to execute on the instance via SSH once initialized |
| `--script-args` | `""` | String arguments to pass to the script |
| `-d`, `--detach` | `False` | Run script in background without waiting / streaming |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |
| `--no-wait-for-setup` | `False` | Do not wait for container disk setup to complete before executing script |
| `--ssh-timeout` | `180` | Max seconds to wait for SSH and setup readiness |

#### RunPod-Specific Options
These options are registered dynamically when targeting RunPod (`xpush --provider runpod create ...`):

| Flag | Default | Description |
|---|---|---|
| `--template-id` | *None* | Pod Template ID to deploy (auto-resolves container image name) |
| `--volume-id` | *None* | Persistent Network Volume ID to mount |
| `--cloud-type` | `SECURE` | Type of cloud network (`SECURE`, `COMMUNITY`, or `ALL`) |
| `--ports` | `22/tcp` | Container ports to expose |
| `--docker-args` | *None* | Optional custom docker arguments |
| `--vcpu-count` | `4` | Minimum number of vCPUs to allocate |
| `--memory` | `8` | Minimum CPU RAM in GB to allocate |

#### Vast.ai-Specific Options
These options are registered dynamically when targeting Vast.ai (`xpush --provider vast create ...`):

| Flag | Default | Description |
|---|---|---|
| `--offer-id` | *None* | Direct Vast.ai offer/ask ID to rent (auto-switches provider to `vast`) |

#### Spheron-Specific Options
These options are registered dynamically when targeting Spheron (`xpush --provider spheron create ...`):

| Flag | Default | Description |
|---|---|---|
| `--offer-id` | *None* | Direct Spheron offer ID to deploy |
| `--ssh-key-id` | *None* | Spheron pre-registered SSH key ID (auto-switches provider to `spheron`) |
| `--region` | *None* | Target deployment region (optional) |

> [!NOTE]
> `xpu-shell` uses provider-driven argument parsing. Provider-specific flags (e.g. `--offer-id` for Vast or `--template-id` for RunPod) are only registered for their respective provider and will be rejected if passed to an incompatible provider. Run `xpush --provider <name> create --help` to inspect provider-specific help.

> [!TIP]
> Creating an instance automatically saves its ID and provider to `~/.config/xpu_shell/.last_pod_id` and `~/.config/xpu_shell/.last_provider`, so subsequent commands (`exec`, `ps`, `logs`, etc.) seamlessly target it without passing any flags.

---

### 2. `list`
Lists active and stopped instances associated with your account, showing Instance ID, Name, Status, GPU type, and SSH connection endpoint.

```bash
# List instances for default/current provider:
xpush list

# Or specify a provider explicitly:
xpush list --provider vast
xpush list --provider runpod
```


---

### 3. `info`
Shows detailed information about a specific pod or the last created/used pod (mimicking the RunPod dashboard "Details" tab). Displays hardware specifications (GPU model & VRAM, allocated vCPUs, RAM size, container & volume disk sizes), live utilization metrics (CPU, RAM, GPU compute & VRAM), uptime, cost, datacenter location, and direct SSH connection strings. Supports `--json` for automation.

```bash
# Show info for the last created pod
xpush info

# Specify a pod explicitly
xpush info <pod-id>
xpush info -p <pod-id>
xpush info --pod <pod-id>

# Output raw JSON
xpush info <pod-id> --json
```

| Argument / Flag | Default | Description |
|---|---|---|
| `pod-id` | *None* | Target pod ID (optional positional, defaults to last created pod) |
| `-p`, `--pod` | *None* | Target pod ID (defaults to last created pod) |
| `--json` | `False` | Output raw pod details in JSON format |

---

### 4. `templates` (RunPod only)
Lists available pod templates associated with your account and public library, showing Template ID, Name, and Image name. Supports optional regex filtering. Only registered when targeting RunPod.

```bash
# List all templates
xpush templates

# Filter templates by pattern (case-insensitive regex)
xpush templates pytorch
xpush templates "vllm|tgi"
xpush templates -r "pytorch|cuda"
```

| Argument / Flag | Default | Description |
|---|---|---|
| `filter` | *None* | Optional regex pattern to filter templates by ID, Name, or Image (case-insensitive) |
| `-r`, `--regex`, `--filter` | *None* | Optional regex pattern to filter templates by ID, Name, or Image (case-insensitive) |

---

### 5. `stop`
Stops a running pod (releases GPU resources, but retains data on the persistent network volume). If `<pod-id>` is omitted, automatically targets the last created pod.

```bash
# Stop the last created pod
xpush stop

# Or specify a pod explicitly
xpush stop <pod-id>
xpush stop -p <pod-id>
xpush stop --pod <pod-id>
```

| Argument / Flag | Default | Description |
|---|---|---|
| `pod-id` | *None* | The ID of the pod to stop (optional positional, defaults to last created pod) |
| `-p`, `--pod` | *None* | The ID of the pod to stop (defaults to last created pod) |

---

### 6. `terminate`
Deletes a pod and releases all associated resources. If `<pod-id>` is omitted, automatically targets the last created pod. Also clears the cached `.last_pod_id` if it matches.

```bash
# Terminate the last created pod
xpush terminate

# Or specify a pod explicitly
xpush terminate <pod-id>
xpush terminate -p <pod-id>
xpush terminate --pod <pod-id>
```

| Argument / Flag | Default | Description |
|---|---|---|
| `pod-id` | *None* | The ID of the pod to terminate (optional positional, defaults to last created pod) |
| `-p`, `--pod` | *None* | The ID of the pod to terminate (defaults to last created pod) |

---

### 7. `gpus`
Retrieves and lists available GPU models, including display names, VRAM sizes, maximum GPU configurations, and hourly pricing. Supports optional regex filtering on GPU ID or Display Name (case-insensitive), as well as `-l` / `--expand` to inspect individual marketplace offerings (with Offer IDs, reliability, download/upload speeds, disk space, and location on Vast.ai).

```bash
# List all GPUs
xpush gpus

# Filter by regex (positional or --regex/-r/--filter flag)
xpush gpus 4090
xpush gpus "A100|H100"
xpush gpus -r "RTX 40\d0"

# Expand individual marketplace offerings on Vast.ai (shows Offer ID, reliability, speeds, etc.)
xpush gpus --provider vast 4090 -l
```

| Argument / Flag | Default | Description |
|---|---|---|
| `filter` | *None* | Optional regex pattern to filter GPUs by ID or Display Name (case-insensitive) |
| `-r`, `--regex`, `--filter` | *None* | Optional regex pattern to filter GPUs by ID or Display Name (case-insensitive) |
| `-l`, `--expand` | `False` | Expand list to show individual marketplace offerings (e.g. for Vast.ai) |

---

### 8. `exec`
Uploads and executes an arbitrary local script on an active instance via SSH. Supports foreground streaming or detached background execution, in-memory environment variable injection, and default instance resolution. Automatically remembers the executed Job ID in `~/.config/xpu_shell/.last_job_id`.

```bash
# Execute on the last created instance
xpush exec <script-path> [OPTIONS]

# Specify instance ID explicitly
xpush exec <instance-id> <script-path> [OPTIONS]
xpush exec -p <instance-id> <script-path> [OPTIONS]
```

| Argument / Flag | Default | Description |
|---|---|---|
| `script-path` | *Required* | Path to the local script to run (or 2nd positional if instance ID is given first) |
| `pod-id` | *None* | Optional target instance ID if passed as the first positional argument |
| `-p`, `--pod` | *None* | Target instance ID (defaults to last created instance) |
| `-e`, `--env` | *None* | Environment variable to inject into the remote process in-memory (`KEY=VALUE` or `KEY` to inherit value from local environment). Can accept multiple variables or be repeated. |
| `--env-file`, `--env_file` | *None* | Path to local `.env` file to inject into the remote process in-memory without persisting credentials to remote disk. Can be specified multiple times. |
| `--script-args` | `""` | String arguments to pass to the script |
| `-d`, `--detach` | `False` | Run script in background without waiting / streaming |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |
| `--no-wait-for-setup` | `False` | Do not wait for container disk setup to complete |
| `--ssh-timeout` | `180` | Max seconds to wait for SSH and setup readiness |
| `-v`, `--verbose` | `False` | Show detailed job information and connection progress |

---

### 9. `run`
Executes an arbitrary command line directly (binary + args) on an active instance via SSH. Supports foreground streaming or detached background execution, in-memory environment variable injection, and default instance resolution. Automatically remembers the executed Job ID in `~/.config/xpu_shell/.last_job_id`.


```bash
# Run command on the last created pod
xpush run python3 -c "print('hello')"
xpush run nvidia-smi

# Pass arguments and options to the remote command
xpush run python train.py --epochs 10 --batch-size 32

# Run via remote shell (enables wildcards, pipes, redirections, and command chaining)
xpush run -s mv '/workspace/temp/co*' /workspace/
xpush run --shell "cat /workspace/data.jsonl | grep ERROR > /workspace/errors.log"

# Run detached in background on a specific pod
xpush run -p <pod-id> -d python train.py
```

| Argument / Flag | Default | Description |
|---|---|---|
| `cmd` | *Required* | Command line to execute directly on the remote pod (binary + args) |
| `-p`, `--pod` | *None* | Target pod ID (defaults to last created pod) |
| `-s`, `--shell` | `False` | Execute command within a remote shell (enables wildcards, pipes, redirections) |
| `-d`, `--detach` | `False` | Run command in background without waiting / streaming |
| `-e`, `--env` | *None* | Environment variable to inject into the remote process in-memory (`KEY=VALUE` or `KEY` to inherit value from local environment). Can accept multiple variables or be repeated. |
| `--env-file`, `--env_file` | *None* | Path to local `.env` file to inject into the remote process in-memory without persisting credentials to remote disk. Can be specified multiple times. |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* (or `$RUNPOD_SSH_CONFIG`) | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |
| `--no-wait-for-setup` | `False` | Do not wait for container disk setup to complete |
| `--ssh-timeout` | `180` | Max seconds to wait for SSH and setup readiness |
| `-v`, `--verbose` | `False` | Show detailed job information and connection progress |

---

### 10. `ps`
Lists remote processes and background jobs managed by `xpush` on the pod, including Job ID, PID, running/completed/failed status, start time, duration, and log file path. If `<pod-id>` is omitted, automatically targets the last created pod.

```bash
# List jobs on the last created pod
xpush ps

# Or specify a pod explicitly
xpush ps <pod-id>
xpush ps -p <pod-id>
xpush ps --pod <pod-id>
```

| Argument / Flag | Default | Description |
|---|---|---|
| `pod-id` | *None* | Target pod ID (optional positional, defaults to last created pod) |
| `-p`, `--pod` | *None* | Target pod ID (defaults to last created pod) |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* (or `$RUNPOD_SSH_CONFIG`) | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |

---

### 11. `logs`
Inspects remote execution logs with full display (`cat`), tailing the last N lines, or live streaming (`-f`). If `-p`/`--pod` or `<pod-id>` is omitted, automatically targets the last created pod. If `-j`/`--job` or `<job-id>` is omitted, automatically targets the last executed job.

```bash
# Follow logs on the last executed job on the default pod
xpush logs -f

# View full logs of a specific job on the default pod
xpush logs -j <job-id>
xpush logs <job-id>

# Show last 100 lines for a specific pod and job
xpush logs -p <pod-id> -j <job-id> -n 100
xpush logs -p <pod-id> -j <job-id> -f
```

| Argument / Flag | Default | Description |
|---|---|---|
| `-j`, `--job` | *None* | Target Job ID or PID (defaults to last executed job ID) |
| `job-id` | *None* | Target Job ID or PID (optional positional; defaults to last executed job ID) |
| `pod-id` | *None* | Target pod ID (optional positional, defaults to last created pod) |
| `-p`, `--pod` | *None* | Target pod ID (defaults to last created pod) |
| `-n`, `--tail` | *None* | Number of lines to display from end of log |
| `-f`, `--follow` | `False` | Follow log output in real-time |
| `-v`, `--verbose` | `False` | Show detailed job information |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* (or `$RUNPOD_SSH_CONFIG`) | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |

---

### 12. `kill`
Terminates a remote job and its entire process group. By default, sends `SIGTERM` first, monitors process termination, and escalates to `SIGKILL` if the job has not exited within `--timeout` seconds. If `-p`/`--pod` or `<pod-id>` is omitted, automatically targets the last created pod. If `-j`/`--job` or `<job-id>` is omitted, automatically targets the last executed job.

```bash
# Kill the last executed job on the last created pod
xpush kill

# Kill a specific job on the default pod
xpush kill -j <job-id-or-pid> [OPTIONS]
xpush kill <job-id-or-pid> [OPTIONS]

# Specify pod ID and job ID explicitly
xpush kill -p <pod-id> -j <job-id-or-pid> [OPTIONS]
```

| Argument / Flag | Default | Description |
|---|---|---|
| `-j`, `--job` | *None* | Target Job ID or PID to kill (defaults to last executed job ID) |
| `job-id-or-pid` | *None* | Target Job ID or PID to kill (optional positional, defaults to last executed job ID) |
| `pod-id` | *None* | Target pod ID (optional positional if specified before target job/PID) |
| `-p`, `--pod` | *None* | Target pod ID (defaults to last created pod) |
| `-s`, `--signal` | `SIGTERM` | Initial signal to send (e.g. `SIGTERM`, `SIGKILL`) |
| `-t`, `--timeout` | `30.0` | Timeout in seconds to wait before escalating from SIGTERM to SIGKILL |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* (or `$RUNPOD_SSH_CONFIG`) | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |

---

### 13. `cp`
Copies files or directories between the local host and an active RunPod instance using `scp`. Supports automatic SSH port and key detection, directory recursion (`-r`), attribute preservation (`-P`), and remote path resolution using `:` or `<pod-id>:`.

Remote paths are distinguished by a colon (`:`):
* `:/remote/path` — Targets the path on the default pod (or the pod specified by `-p`/`--pod`).
* `<pod_id>:/remote/path` — Targets the path on a specific pod ID.
* Local paths contain no colon.

```bash
# Copy local file to the default pod's /workspace
xpush cp myfile.tar.gz :/workspace/

# Copy a file from a specific pod to the current local directory
xpush cp <pod-id>:/workspace/results.csv ./

# Recursively copy a remote directory to local
xpush cp -r :/workspace/checkpoints/ ./checkpoints/

# Copy matching files using remote wildcards (quote to prevent local expansion)
xpush cp ':/workspace/temp/co*' ./
```

| Argument / Flag | Default | Description |
|---|---|---|
| `paths` | *Required* | Source and destination paths (e.g. `local.txt :/workspace/` or `:/workspace/data.csv ./`) |
| `-p`, `--pod` | *None* | Target pod ID (defaults to last created pod if not specified in path prefix) |
| `-r`, `-R`, `--recursive` | `False` | Recursively copy entire directories |
| `-P`, `--preserve` | `False` | Preserves modification times, access times, and modes from original file |
| `-q`, `--quiet` | `False` | Quiet mode: disables progress meter and non-fatal messages |
| `--ssh-private-key-path` | *None* | Path to private SSH key (auto-detected if omitted) |
| `--ssh-config` | *None* (or `$RUNPOD_SSH_CONFIG`) | Path to custom SSH config file (e.g. `/dev/null`, or `system`) |

---

### Examples

### Launch on RunPod (default)
```bash
xpush create
# The newly created instance ID is automatically remembered!
```

### Launch on Vast.ai
```bash
# Search available GPU offers and pricing on Vast.ai:
xpush gpus --provider vast -r 4090

# Auto-rent the cheapest available RTX 4090 offer on Vast.ai:
xpush create --provider vast --gpu-type 4090

# Or rent a specific Vast.ai offer ID directly:
xpush create --provider vast --offer-id 1234567
```

### Launch using a Template ID (RunPod)
```bash
# Search for templates
xpush templates pytorch

# Create pod from template
xpush create --template-id "runpod-pytorch" --gpu-type 4090
```

### Launch attaching a Network Volume and Python requirements
```bash
xpush create --volume-id "vol-abc123xyz" --requirements-path requirements.txt
```

### Run commands or scripts with in-memory secrets and monitor jobs
```bash
# Execute a command directly in the background with in-memory credentials:
# (Works seamlessly on both RunPod and Vast.ai!)
xpush run -d -e R2_TOKEN R2_ACCESS_KEY_ID python train.py --epochs 50

# Or upload and run a local script
xpush exec -d ./train.py

# Check process status on default instance
xpush ps

# Follow logs in real-time (automatically targets the last job ID on default instance)
xpush logs -f

# Or view logs for a specific job explicitly
xpush logs -j job-1788613220-1817e4 -f

# Terminate gracefully (automatically targets the last job ID, or specify with -j)
xpush kill
```

### Fully customized creation
```bash
xpush create \
  --image-name "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-ubuntu22.04" \
  --gpu-type "NVIDIA RTX A6000" \
  --gpu-count 2 \
  --env-file secrets.env \
  --pip-packages torchinfo matplotlib
```

### Running inside a Bubblewrap Sandbox or Container
When running inside unprivileged user namespaces or Bubblewrap sandboxes, host system files under `/etc/ssh/ssh_config.d/` can trigger OpenSSH `Bad owner or permissions` errors. You can bypass them either globally:

```bash
export RUNPOD_SSH_CONFIG="/dev/null"
xpush exec ./train.py
```

Or per-command using `--ssh-config`:

```bash
xpush exec ./train.py --ssh-config /dev/null
```

