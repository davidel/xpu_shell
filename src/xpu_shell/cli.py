import argparse
import base64
from dataclasses import asdict
from datetime import datetime, timezone
import difflib
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import runpod

from xpu_shell.ssh_runner import (
    find_ssh_private_key,
    resolve_ssh_config,
    execute_remote_script,
    execute_remote_command,
    list_remote_jobs,
    view_remote_logs,
    kill_remote_job,
    wait_for_ssh,
    build_ssh_cmd,
    build_cp_cmd,
    build_rsync_cmd
)
from xpu_shell.providers import (
    BaseProvider,
    get_provider,
    resolve_provider,
    save_last_provider,
    get_last_provider,
    parse_instance_identifier,
    parse_template_identifier,
    format_template_identifier
)
from xpu_shell.proxy_manager import (
    start_proxy_for_pod,
    stop_proxy_for_pod,
    get_active_proxy_info,
    list_active_proxies
)


class ParseEnv(argparse.Action):

  def __call__(self, parser, namespace, values, option_string=None):
    env_dict = getattr(namespace, self.dest)
    if env_dict is None or not isinstance(env_dict, dict):
      env_dict = {}
    if isinstance(values, str):
      values = [values]
    for val in values:
      if '=' in val:
        k, v = val.split('=', 1)
        k = k.strip()
        if not k:
          parser.error(f"Invalid environment variable format: '{val}'. Key cannot be empty.")
        env_dict[k] = v
      else:
        k = val.strip()
        if not k:
          parser.error(f"Invalid environment variable format: '{val}'. Key cannot be empty.")
        if k in os.environ:
          env_dict[k] = os.environ[k]
        else:
          parser.error(f"Environment variable '{k}' is not set in the local environment.")
    setattr(namespace, self.dest, env_dict)


CONFIG_DIR = Path.home() / ".config" / "xpu_shell"
LAST_POD_ID_FILE = CONFIG_DIR / ".last_pod_id"
LAST_JOB_ID_FILE = CONFIG_DIR / ".last_job_id"


def get_config_dir():
  custom_dir = os.environ.get("XPU_SHELL_CONFIG_DIR") or os.environ.get("RUNPOD_SHELL_CONFIG_DIR")
  if custom_dir:
    return Path(custom_dir)
  return CONFIG_DIR


def get_last_pod_id_file():
  custom_file = os.environ.get("XPU_SHELL_LAST_POD_ID_FILE")
  if custom_file:
    return Path(custom_file)
  if "XPU_SHELL_CONFIG_DIR" in os.environ or "RUNPOD_SHELL_CONFIG_DIR" in os.environ:
    return get_config_dir() / ".last_pod_id"
  return LAST_POD_ID_FILE


def save_last_pod_id(pod_id, provider=None):
  try:
    target_file = get_last_pod_id_file()
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(str(pod_id).strip())
    if provider:
      save_last_provider(provider)
  except Exception:
    pass


def get_last_pod_id():
  try:
    target_file = get_last_pod_id_file()
    if target_file.exists():
      pid = target_file.read_text().strip()
      if pid:
        return pid
  except Exception:
    pass
  return None


def clear_last_pod_id_if_match(pod_id):
  try:
    target_file = get_last_pod_id_file()
    if target_file.exists() and target_file.read_text().strip() == str(pod_id).strip():
      target_file.unlink(missing_ok=True)
  except Exception:
    pass


def get_last_job_id_file():
  custom_file = os.environ.get("XPU_SHELL_LAST_JOB_ID_FILE")
  if custom_file:
    return Path(custom_file)
  if "XPU_SHELL_CONFIG_DIR" in os.environ:
    return get_config_dir() / ".last_job_id"
  return LAST_JOB_ID_FILE


def save_last_job_id(job_id):
  try:
    target_file = get_last_job_id_file()
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(str(job_id).strip())
  except Exception:
    pass


def get_last_job_id():
  try:
    target_file = get_last_job_id_file()
    if target_file.exists():
      jid = target_file.read_text().strip()
      if jid:
        return jid
  except Exception:
    pass
  return None


def clear_last_job_id_if_match(job_id):
  try:
    target_file = get_last_job_id_file()
    if target_file.exists() and target_file.read_text().strip() == str(job_id).strip():
      target_file.unlink(missing_ok=True)
  except Exception:
    pass




def resolve_pod_id(args):
  pod_id = getattr(args, "pod", None) or getattr(args, "pod_id", None)
  if not pod_id:
    pod_id = get_last_pod_id()
  if not pod_id:
    fatal(
        f"No pod ID specified and no previous pod ID found in {get_last_pod_id_file()}. "
        "Please specify -p/--pod <pod_id>."
    )
  return pod_id


def resolve_job_id(args, required=True):
  job_id = getattr(args, "job", None)
  if job_id:
    save_last_job_id(job_id)
  else:
    job_id = get_last_job_id()
  if not job_id and required:
    fatal(
        f"No job ID specified and no previous job ID found in {get_last_job_id_file()}. "
        "Please specify -j/--job <job_id>."
    )
  return job_id


def preprocess_argv(argv):
  if not argv:
    return argv
  try:
    exec_idx = argv.index("exec")
  except ValueError:
    return argv

  prefix = argv[:exec_idx + 1]
  sub_args = argv[exec_idx + 1:]

  flags_0 = {"-d", "--detach", "--no-wait-for-setup", "-v", "--verbose", "-h", "--help"}
  opts_1 = {
      "-p",
      "--pod",
      "-j",
      "--job",
      "--script-args",
      "--ssh-private-key-path",
      "--ssh-config",
      "--ssh-timeout",
      "--env-file",
      "--env_file"
  }

  has_pod_opt = any(a in ("-p", "--pod") or a.startswith("--pod=") or a.startswith("-p=") for a in sub_args)
  has_env_flag = any(a in ("-e", "--env") for a in sub_args)
  if not has_env_flag:
    return argv

  new_sub = []
  i = 0
  n = len(sub_args)
  env_tokens = []
  standalone_positionals = []

  while i < n:
    arg = sub_args[i]
    if arg in flags_0:
      new_sub.append(arg)
      i += 1
    elif any(arg == opt or arg.startswith(opt + "=") for opt in opts_1):
      new_sub.append(arg)
      if "=" not in arg and i + 1 < n and not sub_args[i + 1].startswith("-"):
        new_sub.append(sub_args[i + 1])
        i += 1
      i += 1
    elif arg in ("-e", "--env"):
      i += 1
      while i < n and not sub_args[i].startswith("-"):
        env_tokens.append(sub_args[i])
        i += 1
    else:
      standalone_positionals.append(arg)
      i += 1

  positionals_from_env = []
  real_env_vars = []

  if has_pod_opt:
    if not standalone_positionals and env_tokens:
      positionals_from_env.append(env_tokens[-1])
      real_env_vars = env_tokens[:-1]
    else:
      real_env_vars = env_tokens
  else:
    if len(standalone_positionals) >= 2:
      real_env_vars = env_tokens
    elif len(standalone_positionals) == 1:
      real_env_vars = env_tokens
    else:
      if env_tokens:
        script_tok = env_tokens[-1]
        remaining = env_tokens[:-1]
        if remaining:
          cand = remaining[-1]
          if "=" not in cand and cand not in os.environ:
            positionals_from_env = [cand, script_tok]
            real_env_vars = remaining[:-1]
          else:
            positionals_from_env = [script_tok]
            real_env_vars = remaining
        else:
          positionals_from_env = [script_tok]
          real_env_vars = []

  result = prefix + new_sub
  for ev in real_env_vars:
    result.extend(["-e", ev])
  result.extend(standalone_positionals)
  result.extend(positionals_from_env)
  return result


def fatal(msg, exc=RuntimeError):
  print(msg, file=sys.stderr)
  raise exc(msg)


def read_ssh_key(key_path):
  with open(key_path, 'r') as f:
    return f.read().strip()


def get_ssh_key(key_path_str=None):
  if key_path_str:
    key_path = Path(key_path_str).expanduser()
    if not key_path.exists():
      fatal(f"SSH key not found at {key_path}", FileNotFoundError)
    return read_ssh_key(key_path)

  # Check default paths in ~/.ssh
  ssh_dir = Path.home() / ".ssh"
  if ssh_dir.exists():
    for name in ["id_rsa.pub", "id_ed25519.pub", "id_ecdsa.pub", "id_dsa.pub"]:
      key_path = ssh_dir / name
      if key_path.exists():
        print(f"Using default SSH key: {key_path}")
        return read_ssh_key(key_path)

  fatal(
      "No SSH key path provided and no default key (id_rsa.pub, id_ed25519.pub, id_ecdsa.pub, id_dsa.pub) found in ~/.ssh/",
      FileNotFoundError
  )


def read_requirements(req_path_str=None):
  if not req_path_str:
    req_path = Path("requirements.txt")
    if not req_path.exists():
      return ""
  else:
    req_path = Path(req_path_str).expanduser()
    if not req_path.exists():
      fatal(f"Requirements file not found at {req_path}", FileNotFoundError)

  with open(req_path, 'r') as f:
    return f.read()


def parse_env_file(file_path_str):
  file_path = Path(file_path_str).expanduser()
  if not file_path.exists():
    fatal(f"Environment file not found at {file_path}", FileNotFoundError)

  env_dict = {}
  with open(file_path, "r") as f:
    for line in f:
      line = line.strip()
      # Skip comments and empty lines
      if not line or line.startswith("#"):
        continue
      if line.startswith("export "):
        line = line[7:].strip()
      if "=" in line:
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if len(v) >= 2 and ((v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'"))):
          v = v[1:-1]
        env_dict[k] = v
  return env_dict


def read_apt_packages_file(file_path_str):
  file_path = Path(file_path_str).expanduser()
  if not file_path.exists():
    fatal(f"Apt packages file not found at {file_path}", FileNotFoundError)

  packages = []
  with open(file_path, "r") as f:
    for line in f:
      line = line.strip()
      # Skip comments and empty lines
      if not line or line.startswith("#"):
        continue
      # Split by whitespace to handle multiple packages on same line if they exist
      packages.extend(line.split())
  return packages


def build_container_setup_script(apt_packages, requirements_content, pip_packages, volume_mount_path):
  apt_packages_str = " ".join(shlex.quote(p) for p in apt_packages) if apt_packages else ""
  pip_packages_str = " ".join(shlex.quote(p) for p in pip_packages) if pip_packages else ""

  lines = [
      "#!/bin/bash",
      "set -eo pipefail",
      "",
      "log() {",
      '  echo "[$(date \'+%Y-%m-%d %H:%M:%S\')] [setup] $*"',
      "}",
      "",
      'log "Starting container environment setup..."',
      f'mkdir -p "{volume_mount_path}" 2>/dev/null || true'
  ]

  if apt_packages_str:
    lines.extend([
        f'log "Updating apt repositories and installing packages: {apt_packages_str}..."',
        f"apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y {apt_packages_str}",
        'log "Apt packages installed successfully."'
    ])

  if requirements_content.strip() or pip_packages_str:
    lines.extend([
        'log "Configuring system pip for container environment..."',
        'mkdir -p /etc',
        'cat <<\'EOF\' > /etc/pip.conf',
        '[global]',
        'break-system-packages = true',
        'EOF',
        'export PIP_BREAK_SYSTEM_PACKAGES=1',
        'if [ -f /etc/environment ] && ! grep -q "PIP_BREAK_SYSTEM_PACKAGES" /etc/environment 2>/dev/null; then',
        '  echo "PIP_BREAK_SYSTEM_PACKAGES=1" >> /etc/environment',
        'fi',
        'if [ -f /root/.profile ] && ! grep -q "PIP_BREAK_SYSTEM_PACKAGES" /root/.profile 2>/dev/null; then',
        '  echo "export PIP_BREAK_SYSTEM_PACKAGES=1" >> /root/.profile',
        'fi',
        'if [ -f /root/.bashrc ] && ! grep -q "PIP_BREAK_SYSTEM_PACKAGES" /root/.bashrc 2>/dev/null; then',
        '  echo "export PIP_BREAK_SYSTEM_PACKAGES=1" >> /root/.bashrc',
        'fi'
    ])

  escaped_requirements = requirements_content.replace("'", "'\\''")
  if requirements_content.strip():
    lines.extend([
        'log "Writing requirements.txt..."',
        f"echo '{escaped_requirements}' > \"{volume_mount_path}/requirements.txt\"",
        f'log "Installing packages from {volume_mount_path}/requirements.txt..."',
        f'pip install -r "{volume_mount_path}/requirements.txt"'
    ])

  if pip_packages_str:
    lines.extend([
        f'log "Installing command-line pip packages: {pip_packages_str}..."',
        f'pip install {pip_packages_str}'
    ])

  lines.extend([
      f'mkdir -p "{volume_mount_path}" 2>/dev/null || true',
      f'touch "{volume_mount_path}/.setup_complete" 2>/dev/null || touch /tmp/.setup_complete',
      'log "Container environment setup finished successfully."'
  ])

  return "\n".join(lines) + "\n"


def run_container_setup(
    host,
    port,
    setup_script_content,
    private_key_path=None,
    detach=False,
    ssh_timeout=180,
    ssh_config_path=None
):
  import tempfile
  with tempfile.NamedTemporaryFile(mode="w", suffix="_setup_container.sh", delete=False) as tf:
    tf.write(setup_script_content)
    temp_path = tf.name

  try:
    print("\nRunning container environment setup via SSH...")
    res = execute_remote_script(
        host=host,
        port=port,
        script_path=temp_path,
        script_args="",
        detach=detach,
        private_key_path=private_key_path,
        wait_for_setup_flag=False,
        ssh_timeout=ssh_timeout,
        ssh_config_path=ssh_config_path
    )
    if not detach and res.get("exit_code", 0) != 0:
      fatal(f"Container setup failed with exit code: {res.get('exit_code')}")
    return res
  finally:
    if os.path.exists(temp_path):
      os.remove(temp_path)


def get_valid_gpus():
  try:
    gpus = runpod.get_gpus()
    return [g.get("id") for g in gpus if g.get("id")]
  except Exception:
    return None


def resolve_gpu_type(user_input, valid_gpus):
  if not valid_gpus:
    return user_input

  # 1. Exact match
  if user_input in valid_gpus:
    return user_input

  # 2. Case-insensitive match
  user_lower = user_input.lower()
  for gpu in valid_gpus:
    if gpu.lower() == user_lower:
      return gpu

  # 3. Substring match
  matches = []
  for gpu in valid_gpus:
    if user_lower in gpu.lower():
      matches.append(gpu)

  if len(matches) == 1:
    print(f"Auto-resolved GPU type '{user_input}' to '{matches[0]}'")
    return matches[0]
  elif len(matches) > 1:
    options_str = ", ".join(f"'{m}'" for m in matches)
    fatal(f"GPU type '{user_input}' is ambiguous. Did you mean one of: {options_str}?", ValueError)

  # 4. Fuzzy match
  close_matches = difflib.get_close_matches(user_input, valid_gpus, n=3, cutoff=0.4)
  if close_matches:
    options_str = ", ".join(f"'{m}'" for m in close_matches)
    fatal(f"GPU type '{user_input}' not found. Did you mean: {options_str}?", ValueError)

  # Fallback
  fatal(f"GPU type '{user_input}' not found and no close matches detected.", ValueError)


def get_pod_ssh_endpoint(pod_info):
  runtime = pod_info.get("runtime", {}) if pod_info else {}
  ports = runtime.get("ports", []) if runtime else []
  ssh_port = None
  ssh_host = None

  if isinstance(ports, list):
    for p in ports:
      if p.get("privatePort") == 22:
        ssh_port = p.get("publicPort") or p.get("isExternal")
        ssh_host = p.get("ip") or p.get("address")
        break

  if not ssh_host and pod_info:
    ssh_host = pod_info.get("ipAddress") or pod_info.get("address")

  if not ssh_port and isinstance(ports, dict):
    ssh_port = ports.get("publicPort") or ports.get("isExternal")

  return ssh_host, ssh_port


def resolve_instance_ssh_endpoint(target_pod_id, args, verbose=False):
  if verbose:
    print(f"Fetching connection details for pod '{target_pod_id}'...", file=sys.stderr)
  provider = resolve_provider(args=args, instance_id=target_pod_id)
  if provider.name == "runpod":
    try:
      pod_info = runpod.get_pod(target_pod_id)
    except Exception as e:
      fatal(f"Failed to fetch pod info: {e}", exc=e.__class__)
    if not pod_info:
      fatal(f"Pod '{target_pod_id}' not found.", FileNotFoundError)
    ssh_host, ssh_port = get_pod_ssh_endpoint(pod_info)
  else:
    try:
      inst = provider.get_instance(target_pod_id)
      ssh_host, ssh_port = inst.ssh_host, inst.ssh_port
    except Exception as e:
      fatal(f"Failed to fetch instance info: {e}", exc=e.__class__)

  if not ssh_host or not ssh_port or ssh_port == "unknown":
    fatal(f"Could not resolve SSH endpoint for pod '{target_pod_id}'. Is the pod running and exposing port 22?")
  return ssh_host, ssh_port


def cmd_create(args):

  # Resolve and validate GPU type
  valid_gpus = get_valid_gpus()
  gpu_type = resolve_gpu_type(args.gpu_type, valid_gpus)

  # Load SSH public key
  ssh_public_key = get_ssh_key(args.ssh_key_path)

  # Read requirements
  requirements_content = read_requirements(args.requirements_path)

  # Build apt packages list
  apt_packages = []
  if args.apt_packages_file:
    apt_packages.extend(read_apt_packages_file(args.apt_packages_file))

  if args.apt_packages is not None:
    apt_packages.extend(args.apt_packages)
  elif not args.apt_packages_file and (args.pip_packages or requirements_content.strip()):
    apt_packages = ["screen", "curl", "htop", "ffmpeg", "git"]

  # Build container disk setup script
  container_disk_setup = build_container_setup_script(
      apt_packages=apt_packages,
      requirements_content=requirements_content,
      pip_packages=args.pip_packages,
      volume_mount_path=args.volume_mount_path
  )

  # Load env file if provided
  env_file_vars = {}
  if args.env_file:
    env_file_vars = parse_env_file(args.env_file)

  # Set environment variables (ensuring SSH public key is present)
  container_env = env_file_vars.copy()
  container_env.update(args.env)
  container_env["PUBLIC_KEY"] = ssh_public_key

  provider = resolve_provider(args=args)
  raw_template = getattr(args, "template", None)
  raw_template_id = getattr(args, "template_id", None)
  raw_image_name = getattr(args, "image_name", None)

  template_id = None
  image_name = None

  if raw_template:
    kind, val = parse_template_identifier(raw_template)
    if kind == "image":
      image_name = val
    elif kind == "template":
      template_id = val
  elif raw_template_id:
    template_id = raw_template_id
  elif raw_image_name:
    image_name = raw_image_name

  if provider.name == "vast":
    print(f"Launching Vast.ai instance '{args.name}'...")
    try:
      disk_total = float(args.container_disk_size) + float(args.volume_size or 0)
      inst = provider.create_instance(
          gpu_name=args.gpu_type,
          image_name=image_name,
          template_id=template_id,
          disk_gb=disk_total,
          env=container_env,
          offer_id=getattr(args, "offer_id", None),
          gpu_count=args.gpu_count
      )
    except Exception as e:
      fatal(f"Failed to create Vast.ai instance: {e}", exc=e.__class__)

    save_last_pod_id(inst.id)
    save_last_provider("vast")

    print("Waiting for instance to initialize...")
    start_poll = time.time()
    poll_timeout = getattr(args, "ssh_timeout", 180)
    if poll_timeout < 300:
      poll_timeout = 300
    ssh_host = None
    ssh_port = None
    last_status = None
    while time.time() - start_poll < poll_timeout:
      try:
        inst = provider.get_instance(inst.id)
      except Exception:
        time.sleep(5)
        continue

      if inst.status in ("TERMINATED", "FAILED", "DEAD"):
        fatal(f"Instance initialization failed with status: {inst.status}")

      if inst.status != last_status:
        last_status = inst.status
        print(f"Instance status: {inst.status}")

      if inst.status == "RUNNING" and inst.ssh_host and inst.ssh_port:
        ssh_host = inst.ssh_host
        ssh_port = inst.ssh_port
        break
      time.sleep(5)
    else:
      fatal(f"Timed out waiting for Vast.ai instance '{inst.id}' to become ready after {poll_timeout} seconds.")
  elif provider.name == "spheron":
    print(f"Launching Spheron instance '{args.name}'...")
    try:
      disk_total = float(args.container_disk_size) + float(args.volume_size or 0)
      inst = provider.create_instance(
          name=args.name,
          gpu_name=args.gpu_type,
          image_name=image_name,
          disk_gb=disk_total,
          env=container_env,
          offer_id=getattr(args, "offer_id", None),
          ssh_key_id=getattr(args, "ssh_key_id", None),
          ssh_public_key=ssh_public_key,
          region=getattr(args, "region", None),
          gpu_count=args.gpu_count
      )
    except Exception as e:
      fatal(f"Failed to create Spheron instance: {e}", exc=e.__class__)

    save_last_pod_id(inst.id)
    save_last_provider("spheron")

    print("Waiting for instance to initialize...")
    start_poll = time.time()
    poll_timeout = getattr(args, "ssh_timeout", 180)
    if poll_timeout < 300:
      poll_timeout = 300
    ssh_host = None
    ssh_port = None
    last_status = None
    while time.time() - start_poll < poll_timeout:
      try:
        inst = provider.get_instance(inst.id)
      except Exception:
        time.sleep(5)
        continue

      if inst.status in ("TERMINATED", "FAILED", "DEAD"):
        fatal(f"Instance initialization failed with status: {inst.status}")

      if inst.status != last_status:
        last_status = inst.status
        print(f"Instance status: {inst.status}")

      if inst.status == "RUNNING" and inst.ssh_host and inst.ssh_port:
        ssh_host = inst.ssh_host
        ssh_port = inst.ssh_port
        break
      time.sleep(5)
    else:
      fatal(f"Timed out waiting for Spheron instance '{inst.id}' to become ready after {poll_timeout} seconds.")
  elif provider.name == "cloudrift":
    print(f"Launching CloudRift instance '{args.name}'...")
    try:
      disk_total = float(args.container_disk_size) + float(args.volume_size or 0)
      inst = provider.create_instance(
          name=args.name,
          gpu_name=args.gpu_type,
          image_name=image_name,
          template=template_id,
          disk_gb=disk_total,
          env=container_env,
          ssh_public_key=ssh_public_key,
          datacenter=getattr(args, "datacenter", None),
          gpu_count=args.gpu_count
      )
    except Exception as e:
      fatal(f"Failed to create CloudRift instance: {e}", exc=e.__class__)

    save_last_pod_id(inst.id)
    save_last_provider("cloudrift")

    print("Waiting for instance to initialize...")
    start_poll = time.time()
    poll_timeout = getattr(args, "ssh_timeout", 180)
    if poll_timeout < 300:
      poll_timeout = 300
    ssh_host = None
    ssh_port = None
    last_status = None
    while time.time() - start_poll < poll_timeout:
      try:
        inst = provider.get_instance(inst.id)
      except Exception:
        time.sleep(5)
        continue

      if inst.status in ("TERMINATED", "FAILED", "DEAD"):
        fatal(f"Instance initialization failed with status: {inst.status}")

      if inst.status != last_status:
        last_status = inst.status
        print(f"Instance status: {inst.status}")

      if inst.status == "RUNNING" and inst.ssh_host and inst.ssh_port:
        ssh_host = inst.ssh_host
        ssh_port = inst.ssh_port
        break
      time.sleep(5)
    else:
      fatal(f"Timed out waiting for CloudRift instance '{inst.id}' to become ready after {poll_timeout} seconds.")
  elif provider.name == "lium":
    print(f"Launching Lium instance '{args.name}'...")
    try:
      disk_total = float(args.container_disk_size) + float(args.volume_size or 0)
      inst = provider.create_instance(
          name=args.name,
          gpu_name=args.gpu_type,
          image_name=image_name,
          template=template_id,
          disk_gb=disk_total,
          env=container_env,
          ssh_public_key=ssh_public_key,
          executor_id=getattr(args, "executor_id", None) or getattr(args, "offer_id", None),
          gpu_count=args.gpu_count
      )
    except Exception as e:
      fatal(f"Failed to create Lium instance: {e}", exc=e.__class__)

    save_last_pod_id(inst.id)
    save_last_provider("lium")

    print("Waiting for instance to initialize...")
    start_poll = time.time()
    poll_timeout = getattr(args, "ssh_timeout", 180)
    if poll_timeout < 300:
      poll_timeout = 300
    ssh_host = None
    ssh_port = None
    last_status = None
    while time.time() - start_poll < poll_timeout:
      try:
        inst = provider.get_instance(inst.id)
      except Exception:
        time.sleep(5)
        continue

      if inst.status in ("TERMINATED", "FAILED", "DEAD"):
        fatal(f"Instance initialization failed with status: {inst.status}")

      if inst.status != last_status:
        last_status = inst.status
        print(f"Instance status: {inst.status}")

      if inst.status == "RUNNING" and inst.ssh_host and inst.ssh_port:
        ssh_host = inst.ssh_host
        ssh_port = inst.ssh_port
        break
      time.sleep(5)
    else:
      fatal(f"Timed out waiting for Lium instance '{inst.id}' to become ready after {poll_timeout} seconds.")
  else:
    # Launch Pod on RunPod
    print(f"Launching RunPod instance '{args.name}'...")
    create_args = {
        "name": args.name,
        "gpu_type_id": gpu_type,
        "gpu_count": args.gpu_count,
        "container_disk_in_gb": args.container_disk_size,
        "volume_in_gb": args.volume_size,
        "volume_mount_path": args.volume_mount_path,
        "ports": args.ports,
        "env": container_env,
        "cloud_type": args.cloud_type,
        "min_vcpu_count": args.vcpu_count,
        "min_memory_in_gb": args.memory
    }

    if getattr(args, "docker_args", None):
      create_args["docker_args"] = args.docker_args

    if template_id:
      create_args["template_id"] = template_id
      template = get_pod_template(template_id)
      if not template or not template.get("imageName"):
        fatal(f"Template '{template_id}' not found or does not specify an image name.", ValueError)
      create_args["image_name"] = template["imageName"]
    elif image_name or provider.default_image:
      create_args["image_name"] = image_name or provider.default_image
    else:
      fatal("Either --template-id or --image-name must be specified.", ValueError)

    if args.volume_id:
      create_args["network_volume_id"] = args.volume_id

    try:
      pod = runpod.create_pod(**create_args)
    except Exception as e:
      fatal(f"Failed to create pod: {e}", exc=e.__class__)

    save_last_pod_id(pod["id"])
    save_last_provider("runpod")


    # Wait for the pod to boot up
    print("Waiting for pod to initialize...")
    start_poll = time.time()
    poll_timeout = getattr(args, "ssh_timeout", 180)
    if poll_timeout < 300:
      poll_timeout = 300
    pod_info = None
    last_status = None

    while time.time() - start_poll < poll_timeout:
      try:
        pod_info = runpod.get_pod(pod["id"])
      except Exception as e:
        print(f"Error polling pod status: {e}")
        time.sleep(5)
        continue

      if not pod_info:
        time.sleep(5)
        continue

      status = pod_info.get("desiredStatus") or pod_info.get("status")
      if status in ("EXITED", "TERMINATED", "FAILED", "DEAD"):
        fatal(f"Pod initialization failed with status: {status}")

      if status != last_status:
        last_status = status
        print(f"Pod status: {status}")

      ssh_host, ssh_port = get_pod_ssh_endpoint(pod_info)
      if status == "RUNNING" and ssh_host and ssh_port and ssh_port != "unknown":
        break

      time.sleep(5)
    else:
      fatal(f"Timed out waiting for pod '{pod['id']}' to become ready after {poll_timeout} seconds.")

    ssh_host, ssh_port = get_pod_ssh_endpoint(pod_info)
    if not ssh_host or not ssh_port or ssh_port == "unknown":
      fatal(f"Could not resolve SSH endpoint for pod '{pod['id']}'.")


  ssh_config = getattr(args, "ssh_config", None)
  resolved_cfg = resolve_ssh_config(ssh_config)
  ssh_config_flag = f"-F {resolved_cfg} " if resolved_cfg else ""

  priv_key = find_ssh_private_key(args.ssh_key_path, getattr(args, "ssh_private_key_path", None))
  wait_for_ssh(
      host=ssh_host,
      port=ssh_port,
      private_key_path=priv_key,
      timeout=getattr(args, "ssh_timeout", 180),
      ssh_config_path=ssh_config
  )

  has_setup = bool(
      args.apt_packages or
      args.apt_packages_file or
      requirements_content.strip() or
      args.pip_packages
  )
  no_wait_setup = getattr(args, "no_wait_for_setup", False)

  if has_setup:
    detach_setup = getattr(args, "detach", False) and not getattr(args, "run_script", None)
    run_container_setup(
        host=ssh_host,
        port=ssh_port,
        setup_script_content=container_disk_setup,
        private_key_path=priv_key,
        detach=detach_setup,
        ssh_timeout=getattr(args, "ssh_timeout", 180),
        ssh_config_path=ssh_config
    )
  else:
    sentinel_cmd = f"mkdir -p {args.volume_mount_path} 2>/dev/null; touch {args.volume_mount_path}/.setup_complete 2>/dev/null || touch /tmp/.setup_complete"
    cmd = build_ssh_cmd(
        ssh_host,
        ssh_port,
        sentinel_cmd,
        private_key_path=priv_key,
        ssh_config_path=ssh_config
    )
    subprocess.run(cmd, capture_output=True, text=True)

  if getattr(args, "run_script", None):
    wait_setup = has_setup and not no_wait_setup
    res = execute_remote_script(
        host=ssh_host,
        port=ssh_port,
        script_path=args.run_script,
        script_args=getattr(args, "script_args", "") or "",
        detach=getattr(args, "detach", False),
        private_key_path=priv_key,
        wait_for_setup_flag=wait_setup,
        ssh_timeout=getattr(args, "ssh_timeout", 180),
        ssh_config_path=ssh_config
    )
    if getattr(args, "detach", False) and res.get("job_id"):
      save_last_job_id(res["job_id"])
    if not getattr(args, "detach", False) and res.get("exit_code", 0) != 0:
      sys.exit(res["exit_code"])

  if getattr(args, "auth_proxy", False):
    created_id = None
    if "inst" in locals():
      created_id = inst.id
    elif "pod" in locals():
      created_id = pod.get("id")
    if created_id:
      try:
        from xpu_shell.proxy_manager import start_proxy_for_pod
        p_info = start_proxy_for_pod(
            pod_id=created_id,
            host=ssh_host,
            port=ssh_port,
            remote_port=getattr(args, "proxy_remote_port", 8080),
            private_key_path=priv_key,
            ssh_config_path=ssh_config,
            allow_write=getattr(args, "proxy_allow_write", None),
            verbose=verbose
        )
        print(f"Auth proxy active on remote port {p_info.get('remote_port')}.")
      except Exception as e:
        print(f"Warning: Failed to auto-start auth proxy: {e}", file=sys.stderr)

  print(f"\nPod is ready! Connect via SSH:")
  print(f"ssh {ssh_config_flag}-p {ssh_port} root@{ssh_host}")


def cmd_exec(args):
  if getattr(args, "maybe_script", None):
    target_pod_id = getattr(args, "pod", None) or args.script_or_pod
    script_path = args.maybe_script
  else:
    target_pod_id = getattr(args, "pod", None) or get_last_pod_id()
    script_path = getattr(args, "script_or_pod", None) or getattr(args, "script", None)

  if not target_pod_id:
    fatal(
        f"No pod ID specified and no previous pod ID found in {get_last_pod_id_file()}. "
        "Please specify -p/--pod <pod_id>."
    )

  verbose = getattr(args, "verbose", False)
  ssh_host, ssh_port = resolve_instance_ssh_endpoint(target_pod_id, args, verbose=verbose)


  priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
  wait_setup = not getattr(args, "no_wait_for_setup", False)

  # Resolve environment variables
  extra_env = {}
  env_files = getattr(args, "env_files", None) or []
  if isinstance(env_files, str):
    env_files = [env_files]
  for ef in env_files:
    if ef:
      extra_env.update(parse_env_file(ef))

  if getattr(args, "env", None):
    extra_env.update(args.env)

  proxy_info = get_active_proxy_info(target_pod_id)
  if proxy_info and proxy_info.get("env"):
    for k, v in proxy_info["env"].items():
      if k not in extra_env:
        extra_env[k] = v

  res = execute_remote_script(
      host=ssh_host,
      port=ssh_port,
      script_path=script_path,
      script_args=getattr(args, "script_args", "") or "",
      detach=getattr(args, "detach", False),
      private_key_path=priv_key,
      wait_for_setup_flag=wait_setup,
      ssh_timeout=getattr(args, "ssh_timeout", 180),
      ssh_config_path=getattr(args, "ssh_config", None),
      extra_env=extra_env if extra_env else None,
      verbose=verbose
  )
  if getattr(args, "detach", False) and res.get("job_id"):
    save_last_job_id(res["job_id"])
  if not getattr(args, "detach", False) and res.get("exit_code", 0) != 0:
    sys.exit(res["exit_code"])


def cmd_run(args):
  target_pod_id = resolve_pod_id(args)

  cmd_tokens = getattr(args, "cmd", None) or []
  if cmd_tokens and cmd_tokens[0] == "--":
    cmd_tokens = cmd_tokens[1:]

  if not cmd_tokens:
    fatal("No command specified. Usage: runpod-shell run [OPTIONS] <command> [args...]")

  verbose = getattr(args, "verbose", False)
  ssh_host, ssh_port = resolve_instance_ssh_endpoint(target_pod_id, args, verbose=verbose)


  priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
  wait_setup = not getattr(args, "no_wait_for_setup", False)

  # Resolve environment variables
  extra_env = {}
  env_files = getattr(args, "env_files", None) or []
  if isinstance(env_files, str):
    env_files = [env_files]
  for ef in env_files:
    if ef:
      extra_env.update(parse_env_file(ef))

  if getattr(args, "env", None):
    extra_env.update(args.env)

  proxy_info = get_active_proxy_info(target_pod_id)
  if proxy_info and proxy_info.get("env"):
    for k, v in proxy_info["env"].items():
      if k not in extra_env:
        extra_env[k] = v

  res = execute_remote_command(
      host=ssh_host,
      port=ssh_port,
      command_args=cmd_tokens,
      detach=getattr(args, "detach", False),
      private_key_path=priv_key,
      wait_for_setup_flag=wait_setup,
      ssh_timeout=getattr(args, "ssh_timeout", 180),
      ssh_config_path=getattr(args, "ssh_config", None),
      extra_env=extra_env if extra_env else None,
      use_shell=getattr(args, "use_shell", False),
      verbose=verbose
  )
  if getattr(args, "detach", False) and res.get("job_id"):
    save_last_job_id(res["job_id"])

  if not getattr(args, "detach", False) and res.get("exit_code", 0) != 0:
    sys.exit(res["exit_code"])


def is_remote_path(path_str):
  if path_str.startswith(":"):
    return True
  if ":" in path_str:
    prefix, _ = path_str.split(":", 1)
    if not any(c in prefix for c in ("/\\")):
      return True
  return False


def cmd_cp(args):
  raw_paths = getattr(args, "paths", None) or []
  if len(raw_paths) < 2:
    fatal("cp requires at least one source and one destination. Usage: runpod-shell cp [OPTIONS] <source>... <dest>")

  sources = raw_paths[:-1]
  dest = raw_paths[-1]

  remote_sources = [s for s in sources if is_remote_path(s)]
  dest_is_remote = is_remote_path(dest)

  if remote_sources and dest_is_remote:
    fatal("Direct remote-to-remote copying between pods is not supported. Please copy to local first.")

  if not remote_sources and not dest_is_remote:
    fatal("At least one source or destination must be a remote pod path (e.g. ':/path' or '<pod_id>:/path').")

  pod_id_from_path = None
  for p in (remote_sources if remote_sources else [dest]):
    prefix, _ = p.split(":", 1)
    if prefix:
      pod_id_from_path = prefix
      break

  target_pod_id = pod_id_from_path or resolve_pod_id(args)

  if not getattr(args, "quiet", False):
    print(f"Fetching connection details for pod '{target_pod_id}'...")
  provider = resolve_provider(args=args, instance_id=target_pod_id)
  if provider.name == "runpod":
    try:
      pod_info = runpod.get_pod(target_pod_id)
    except Exception as e:
      fatal(f"Failed to fetch pod info: {e}", exc=e.__class__)

    if not pod_info:
      fatal(f"Pod '{target_pod_id}' not found.", FileNotFoundError)

    ssh_host, ssh_port = get_pod_ssh_endpoint(pod_info)
  else:
    try:
      inst = provider.get_instance(target_pod_id)
      ssh_host, ssh_port = inst.ssh_host, inst.ssh_port
    except Exception as e:
      fatal(f"Failed to fetch instance info: {e}", exc=e.__class__)

  if not ssh_host or not ssh_port or ssh_port == "unknown":
    fatal(f"Could not resolve SSH endpoint for pod '{target_pod_id}'. Is the pod running and exposing port 22?")


  priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
  ssh_cfg = resolve_ssh_config(getattr(args, "ssh_config", None))

  transformed_sources = []
  for s in sources:
    if is_remote_path(s):
      _, rem_path = s.split(":", 1)
      transformed_sources.append(f"root@{ssh_host}:{rem_path}")
    else:
      transformed_sources.append(s)

  if dest_is_remote:
    _, rem_path = dest.split(":", 1)
    transformed_dest = f"root@{ssh_host}:{rem_path}"
  else:
    transformed_dest = dest

  use_rsync = getattr(args, "rsync", False) or getattr(args, "command", "") == "rsync"
  if use_rsync:
    import shutil
    if not shutil.which("rsync"):
      fatal("Host 'rsync' command not found. Please ensure rsync is installed on your system.")

    copy_cmd = build_rsync_cmd(
        sources=transformed_sources,
        dest=transformed_dest,
        port=ssh_port,
        recursive=getattr(args, "recursive", False),
        archive=getattr(args, "archive", False),
        preserve=getattr(args, "preserve", False),
        compress=getattr(args, "compress", True),
        delete=getattr(args, "delete", False),
        exclude=getattr(args, "exclude", None),
        dry_run=getattr(args, "dry_run", False),
        quiet=getattr(args, "quiet", False),
        verbose=getattr(args, "verbose", False),
        private_key_path=priv_key,
        ssh_config_path=ssh_cfg
    )
    cmd_name = "rsync"
  else:
    copy_cmd = build_cp_cmd(
        sources=transformed_sources,
        dest=transformed_dest,
        port=ssh_port,
        recursive=getattr(args, "recursive", False),
        preserve=getattr(args, "preserve", False),
        quiet=getattr(args, "quiet", False),
        private_key_path=priv_key,
        ssh_config_path=ssh_cfg
    )
    cmd_name = "scp"

  try:
    res = subprocess.run(copy_cmd)
    if res.returncode != 0:
      sys.exit(res.returncode)
  except FileNotFoundError:
    fatal(f"Host '{cmd_name}' command not found. Please ensure {cmd_name} is installed on your system.")


def cmd_ps(args):
  target_pod_id = resolve_pod_id(args)
  print(f"Checking processes on pod '{target_pod_id}'...")
  provider = resolve_provider(args=args, instance_id=target_pod_id)
  if provider.name == "runpod":
    try:
      pod_info = runpod.get_pod(target_pod_id)
    except Exception as e:
      fatal(f"Failed to fetch pod info: {e}", exc=e.__class__)

    if not pod_info:
      fatal(f"Pod '{target_pod_id}' not found.", FileNotFoundError)

    ssh_host, ssh_port = get_pod_ssh_endpoint(pod_info)
  else:
    try:
      inst = provider.get_instance(target_pod_id)
      ssh_host, ssh_port = inst.ssh_host, inst.ssh_port
    except Exception as e:
      fatal(f"Failed to fetch instance info: {e}", exc=e.__class__)

  if not ssh_host or not ssh_port or ssh_port == "unknown":
    fatal(f"Could not resolve SSH endpoint for pod '{target_pod_id}'.")


  priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
  jobs = list_remote_jobs(
      ssh_host,
      ssh_port,
      private_key_path=priv_key,
      ssh_config_path=getattr(args, "ssh_config", None)
  )

  if not jobs:
    print(f"No managed jobs found on pod '{target_pod_id}'.")
    return

  jid_width = max(8, max((len(str(j.get("job_id", ""))) for j in jobs), default=8))
  sep_width = max(100, jid_width + 85)
  print(f"{'JOB ID':<{jid_width}} | {'PID':<8} | {'STATUS':<12} | {'STARTED':<20} | {'DURATION':<10} | {'SCRIPT':<18} | {'LOG FILE'}")
  print("-" * sep_width)
  for j in jobs:
    jid = j.get("job_id", "N/A")
    pid = str(j.get("pid", "N/A"))
    status = j.get("status", "N/A")
    started = j.get("started_at_iso", "N/A")
    dur = j.get("duration", "N/A")
    script = j.get("script", "N/A")
    log_f = j.get("log_file", "N/A")
    print(f"{jid:<{jid_width}} | {pid:<8} | {status:<12} | {started:<20} | {dur:<10} | {script:<18} | {log_f}")


def cmd_logs(args):
  explicit_job = getattr(args, "job", None)
  if getattr(args, "pod", None):
    target_pod_id = args.pod
    job_id = explicit_job or getattr(args, "arg1", None)
  elif getattr(args, "arg2", None):
    target_pod_id = args.arg1
    job_id = explicit_job or args.arg2
  elif getattr(args, "arg1", None):
    if explicit_job:
      target_pod_id = args.arg1
      job_id = explicit_job
    else:
      last_id = get_last_pod_id()
      if last_id and (args.arg1.startswith("job-") or args.arg1.isdigit()):
        target_pod_id = last_id
        job_id = args.arg1
      else:
        target_pod_id = args.arg1
        job_id = None
  else:
    target_pod_id = get_last_pod_id()
    job_id = explicit_job

  if job_id:
    save_last_job_id(job_id)
  else:
    job_id = get_last_job_id()

  if not target_pod_id:
    fatal(
        f"No pod ID specified and no previous pod ID found in {get_last_pod_id_file()}. "
        "Please specify -p/--pod <pod_id>."
    )

  verbose = getattr(args, "verbose", False)
  ssh_host, ssh_port = resolve_instance_ssh_endpoint(target_pod_id, args, verbose=verbose)

  priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
  view_remote_logs(
      host=ssh_host,
      port=ssh_port,
      job_id=job_id,
      tail_lines=args.tail,
      follow=args.follow,
      private_key_path=priv_key,
      ssh_config_path=getattr(args, "ssh_config", None),
      verbose=getattr(args, "verbose", False)
  )


def cmd_kill(args):
  explicit_job = getattr(args, "job", None)
  target_pod_id = getattr(args, "pod", None)
  target = explicit_job

  if getattr(args, "maybe_target", None):
    target_pod_id = target_pod_id or args.target_or_pod
    target = target or args.maybe_target
  elif getattr(args, "target_or_pod", None):
    if target:
      target_pod_id = target_pod_id or args.target_or_pod
    else:
      target = args.target_or_pod

  if not target:
    target = get_last_job_id()

  if not target:
    fatal(
        f"No job ID specified and no previous job ID found in {get_last_job_id_file()}. "
        "Please specify -j/--job <job_id>."
    )

  if not target_pod_id:
    target_pod_id = get_last_pod_id()

  if not target_pod_id:
    fatal(
        f"No pod ID specified and no previous pod ID found in {get_last_pod_id_file()}. "
        "Please specify -p/--pod <pod_id>."
    )

  verbose = getattr(args, "verbose", False)
  ssh_host, ssh_port = resolve_instance_ssh_endpoint(target_pod_id, args, verbose=verbose)


  priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
  kill_remote_job(
      host=ssh_host,
      port=ssh_port,
      target_id=target,
      signal_name=args.signal,
      timeout=getattr(args, "timeout", 30.0),
      private_key_path=priv_key,
      ssh_config_path=getattr(args, "ssh_config", None)
  )
  clear_last_job_id_if_match(target)


def cmd_list(args):
  provider = resolve_provider(args=args)
  if provider.name == "runpod":
    print("Listing your RunPod instances...")
    try:
      pods = runpod.get_pods()
    except Exception as e:
      fatal(f"Failed to retrieve pods: {e}", exc=e.__class__)

    if not pods:
      print("No pods found.")
      return

    # Print a nice table
    print(f"{'POD ID':<20} | {'NAME':<25} | {'STATUS':<12} | {'GPU TYPE':<22} | {'SSH ENDPOINT'}")
    print("-" * 100)
    for p in pods:
      pod_id = p.get("id", "N/A")
      name = p.get("name", "N/A")
      status = p.get("desiredStatus") or p.get("status", "N/A")

      # Extract GPU info
      machine_info = p.get("machine") or {}
      gpu_type = (
          machine_info.get("gpuDisplayName")
          or p.get("gpuDisplayName")
          or p.get("gpuName")
          or p.get("gpuTypeId")
          or "GPU"
      )
      gpu_count = p.get("gpuCount", 0)
      if gpu_count > 0:
        gpu_display = f"{gpu_count}x {gpu_type}"
      else:
        gpu_display = "CPU only"

      # Extract SSH endpoint
      host, port = get_pod_ssh_endpoint(p)
      ssh_endpoint = f"{host}:{port}" if host and port and port != "unknown" else "N/A"

      print(f"{pod_id:<20} | {name:<25} | {status:<12} | {gpu_display:<22} | {ssh_endpoint}")
  else:
    print(f"Listing your {provider.name.capitalize()} instances...")
    try:
      instances = provider.list_instances()
    except Exception as e:
      fatal(f"Failed to retrieve instances: {e}", exc=e.__class__)

    if not instances:
      print("No instances found.")
      return

    print(f"{'INSTANCE ID':<20} | {'NAME':<25} | {'STATUS':<12} | {'GPU TYPE':<22} | {'SSH ENDPOINT'}")
    print("-" * 100)
    for inst in instances:
      gpu_display = f"{inst.gpu_count}x {inst.gpu_name}" if inst.gpu_name else "N/A"
      ssh_endpoint = f"{inst.ssh_host}:{inst.ssh_port}" if inst.ssh_host and inst.ssh_port else "N/A"
      print(f"{inst.id:<20} | {inst.name:<25} | {inst.status:<12} | {gpu_display:<22} | {ssh_endpoint}")



def run_graphql(query):
  from runpod.api.graphql import run_graphql_query
  return run_graphql_query(query)


def format_uptime(seconds):
  if seconds is None or seconds < 0:
    return "N/A"
  seconds = int(seconds)
  days = seconds // 86400
  hours = (seconds % 86400) // 3600
  minutes = (seconds % 3600) // 60
  secs = seconds % 60
  parts = []
  if days > 0:
    parts.append(f"{days}d")
  if hours > 0 or days > 0:
    parts.append(f"{hours}h")
  if minutes > 0 or hours > 0 or days > 0:
    parts.append(f"{minutes}m")
  parts.append(f"{secs}s")
  return " ".join(parts)


def get_gpu_vram_map():
  try:
    gpus = runpod.get_gpus()
    vram_map = {}
    for g in gpus:
      mem = g.get("memoryInGb")
      if mem is not None:
        if g.get("id"):
          vram_map[g["id"].lower()] = mem
        if g.get("displayName"):
          vram_map[g["displayName"].lower()] = mem
    return vram_map
  except Exception:
    return {}


def get_detailed_pod(pod_id):
  query = f"""
  query PodDetails {{
    pod(input: {{podId: "{pod_id}"}}) {{
      id
      name
      desiredStatus
      lastStatusChange
      createdAt
      uptimeSeconds
      costPerHr
      podType
      imageName
      dockerArgs
      dockerId
      env
      vcpuCount
      memoryInGb
      containerDiskInGb
      volumeInGb
      volumeMountPath
      gpuCount
      machineId
      ports
      runtime {{
        uptimeInSeconds
        ports {{
          ip
          isIpPublic
          privatePort
          publicPort
          type
        }}
        container {{
          cpuPercent
          memoryPercent
        }}
        gpus {{
          id
          gpuUtilPercent
          memoryUtilPercent
        }}
      }}
      machine {{
        gpuDisplayName
        gpuTypeId
        podHostId
        location
        dataCenterId
      }}
    }}
  }}
  """
  try:
    res = run_graphql(query)
    pod = res.get("data", {}).get("pod")
    if pod:
      return pod
  except Exception:
    pass

  try:
    return runpod.get_pod(pod_id)
  except Exception as e:
    fatal(f"Failed to fetch pod info: {e}", exc=e.__class__)


def cmd_info(args):
  target_pod_id = resolve_pod_id(args)
  provider = resolve_provider(args=args, instance_id=target_pod_id)

  if provider.name == "runpod":
    pod_info = get_detailed_pod(target_pod_id)

    if not pod_info:
      fatal(f"Pod '{target_pod_id}' not found.", FileNotFoundError)

    save_last_pod_id(target_pod_id, provider="runpod")

    if getattr(args, "json_output", False):
      import json
      print(json.dumps(pod_info, indent=2))
      return

    name = pod_info.get("name") or "N/A"
    status = pod_info.get("desiredStatus") or pod_info.get("status") or "N/A"
    pod_type = pod_info.get("podType") or "N/A"
    cost = pod_info.get("costPerHr")
    cost_str = f"${cost:.2f} / hr" if isinstance(cost, (int, float)) else "N/A"

    runtime = pod_info.get("runtime") or {}
    uptime_sec = runtime.get("uptimeInSeconds")
    if uptime_sec is None or uptime_sec == 0:
      uptime_sec = pod_info.get("uptimeSeconds")
    uptime_str = format_uptime(uptime_sec)

    created_at = pod_info.get("createdAt") or pod_info.get("lastStatusChange") or ""
    if created_at and "T" in created_at:
      created_display = created_at.replace("T", " ").split(".")[0] + " UTC"
    else:
      created_display = created_at

    machine = pod_info.get("machine") or {}
    location = machine.get("location")
    datacenter = machine.get("dataCenterId")
    loc_str = datacenter if datacenter else (location if location else "N/A")
    if datacenter and location and datacenter != location:
      loc_str = f"{datacenter} ({location})"

    gpu_count = pod_info.get("gpuCount", 0)
    gpu_name = (
        machine.get("gpuDisplayName")
        or pod_info.get("gpuDisplayName")
        or pod_info.get("gpuName")
        or machine.get("gpuTypeId")
        or "GPU"
    )
    gpu_type_id = machine.get("gpuTypeId") or pod_info.get("gpuTypeId") or ""

    vram_map = get_gpu_vram_map()
    vram_per_gpu = vram_map.get(gpu_type_id.lower()) or vram_map.get(gpu_name.lower())

    if gpu_count > 0:
      if vram_per_gpu:
        total_vram = vram_per_gpu * gpu_count
        if gpu_count == 1:
          gpu_display = f"1x {gpu_name} ({vram_per_gpu} GB VRAM)"
        else:
          gpu_display = f"{gpu_count}x {gpu_name} ({vram_per_gpu} GB each, {total_vram} GB total VRAM)"
      else:
        gpu_display = f"{gpu_count}x {gpu_name}"
    else:
      gpu_display = "CPU only (no dedicated GPU)"

    vcpu = pod_info.get("vcpuCount")
    vcpu_str = f"{vcpu} vCPUs" if vcpu else "N/A"

    ram = pod_info.get("memoryInGb")
    ram_str = f"{ram} GB" if ram else "N/A"

    c_disk = pod_info.get("containerDiskInGb")
    c_disk_str = f"{c_disk} GB" if c_disk else "N/A"

    v_disk = pod_info.get("volumeInGb")
    v_mount = pod_info.get("volumeMountPath") or "/workspace"
    v_disk_str = f"{v_disk} GB (mounted at {v_mount})" if v_disk else "None"

    container_stats = runtime.get("container") or {}
    cpu_pct = container_stats.get("cpuPercent")
    mem_pct = container_stats.get("memoryPercent")
    runtime_gpus = runtime.get("gpus") or []
    gpu_pct = None
    gpu_mem_pct = None
    if runtime_gpus and isinstance(runtime_gpus, list):
      gpu_pct = runtime_gpus[0].get("gpuUtilPercent")
      gpu_mem_pct = runtime_gpus[0].get("memoryUtilPercent")

    host, port = get_pod_ssh_endpoint(pod_info)
    if host and port and port != "unknown":
      ssh_endpoint = f"{host}:{port}"
      ssh_cmd = f"ssh -p {port} root@{host}"
    else:
      ssh_endpoint = "N/A"
      ssh_cmd = "N/A"

    image = pod_info.get("imageName") or "N/A"

    print("=" * 80)
    print(f"Pod Information: {name} ({target_pod_id})")
    print("=" * 80)
    print(f"Status:         {status} ({pod_type})")
    if created_display:
      print(f"Uptime:         {uptime_str} (created: {created_display})")
    else:
      print(f"Uptime:         {uptime_str}")
    print(f"Cost:           {cost_str}")
    print(f"Datacenter:     {loc_str}")
    print()
    print("-- Hardware & Resources " + "-" * 56)
    print(f"GPU:            {gpu_display}")
    print(f"CPU:            {vcpu_str}")
    print(f"Memory (RAM):   {ram_str}")
    print(f"Container Disk: {c_disk_str}")
    print(f"Network Volume: {v_disk_str}")

    has_live_metrics = (cpu_pct is not None or mem_pct is not None or gpu_pct is not None or gpu_mem_pct is not None)
    if has_live_metrics:
      print()
      print("-- Live Utilization " + "-" * 60)
      if cpu_pct is not None:
        print(f"CPU Usage:      {cpu_pct}%")
      if mem_pct is not None:
        print(f"RAM Usage:      {mem_pct}%")
      if gpu_pct is not None:
        print(f"GPU Compute:    {gpu_pct}%")
      if gpu_mem_pct is not None:
        print(f"GPU VRAM:       {gpu_mem_pct}%")

    print()
    print("-- Network & Access " + "-" * 60)
    print(f"SSH Endpoint:   {ssh_endpoint}")
    print(f"SSH Command:    {ssh_cmd}")
    print(f"Docker Image:   {image}")
    print("=" * 80)
  else:
    try:
      inst = provider.get_instance(target_pod_id)
    except Exception as e:
      fatal(f"Instance '{target_pod_id}' not found: {e}", exc=e.__class__)

    if not inst:
      fatal(f"Instance '{target_pod_id}' not found.", FileNotFoundError)

    save_last_pod_id(target_pod_id, provider=provider.name)

    raw_data = inst.raw if (inst.raw and isinstance(inst.raw, dict)) else asdict(inst)

    if getattr(args, "json_output", False):
      import json
      print(json.dumps(raw_data, indent=2))
      return

    name = inst.name or f"{provider.name}-{target_pod_id}"
    status = inst.status or "UNKNOWN"
    raw = inst.raw if isinstance(inst.raw, dict) else {}
    pod_type = (
        raw.get("podType")
        or ("INTERRUPTIBLE" if raw.get("is_bid") else "ON-DEMAND" if "is_bid" in raw else None)
        or raw.get("type")
        or provider.name.upper()
    )

    cost = inst.cost_per_hr
    cost_str = f"${cost:.2f} / hr" if isinstance(cost, (int, float)) else "N/A"

    uptime_sec = inst.uptime_seconds
    if uptime_sec is None:
      duration = raw.get("duration")
      if duration is not None:
        try:
          uptime_sec = int(float(duration))
        except (ValueError, TypeError):
          pass
      elif raw.get("uptime_mins"):
        try:
          uptime_sec = int(float(raw.get("uptime_mins")) * 60)
        except (ValueError, TypeError):
          pass
    uptime_str = format_uptime(uptime_sec)

    created_at = raw.get("createdAt") or raw.get("created_at") or raw.get("start_date") or ""
    created_display = ""
    if isinstance(created_at, (int, float)):
      try:
        created_display = datetime.fromtimestamp(created_at, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
      except Exception:
        created_display = str(created_at)
    elif isinstance(created_at, str) and created_at:
      if "T" in created_at:
        created_display = created_at.replace("T", " ").split(".")[0] + " UTC"
      else:
        created_display = created_at

    location = raw.get("geolocation") or raw.get("datacenter") or raw.get("region") or raw.get("location") or "N/A"

    gpu_count = inst.gpu_count or 0
    gpu_name = inst.gpu_name or raw.get("gpu_name") or raw.get("gpuType") or "GPU"
    vram_map = get_gpu_vram_map()
    vram_per_gpu = vram_map.get(gpu_name.lower())
    if not vram_per_gpu:
      gpu_ram = raw.get("gpu_ram") or raw.get("memoryInGb") or raw.get("gpu_memory")
      if gpu_ram is not None:
        try:
          ram_val = float(gpu_ram)
          vram_per_gpu = round(ram_val / 1024) if ram_val > 256 else round(ram_val)
        except (ValueError, TypeError):
          pass

    if gpu_count > 0:
      if vram_per_gpu:
        total_vram = vram_per_gpu * gpu_count
        if gpu_count == 1:
          gpu_display = f"1x {gpu_name} ({vram_per_gpu} GB VRAM)"
        else:
          gpu_display = f"{gpu_count}x {gpu_name} ({vram_per_gpu} GB each, {total_vram} GB total VRAM)"
      else:
        gpu_display = f"{gpu_count}x {gpu_name}"
    else:
      gpu_display = "CPU only (no dedicated GPU)"

    vcpu = inst.vcpu_count or raw.get("cpu_cores") or raw.get("vcpu")
    cpu_name = raw.get("cpu_name")
    if cpu_name and vcpu:
      vcpu_str = f"{vcpu} vCPUs ({cpu_name})"
    elif vcpu:
      vcpu_str = f"{vcpu} vCPUs"
    else:
      vcpu_str = "N/A"

    ram = inst.memory_gb
    if ram is not None:
      ram_str = f"{ram:.1f} GB" if isinstance(ram, float) and not ram.is_integer() else f"{int(ram)} GB"
    else:
      ram_str = "N/A"

    disk = inst.disk_gb
    if disk is not None:
      disk_str = f"{disk:.1f} GB" if isinstance(disk, float) and not disk.is_integer() else f"{int(disk)} GB"
    else:
      disk_str = "N/A"

    v_disk_str = "None"

    cpu_pct = raw.get("cpu_util")
    mem_pct = raw.get("mem_usage") or raw.get("mem_util")
    gpu_pct = raw.get("gpu_util")
    gpu_mem_pct = raw.get("vmem_usage") or raw.get("vram_usage") or raw.get("gpu_mem_util")

    ssh_host = inst.ssh_host
    ssh_port = inst.ssh_port
    if ssh_host and ssh_port and ssh_port != "unknown":
      ssh_endpoint = f"{ssh_host}:{ssh_port}"
      ssh_cmd = f"ssh -p {ssh_port} root@{ssh_host}"
    else:
      ssh_endpoint = "N/A"
      ssh_cmd = "N/A"

    image = inst.image_name or raw.get("image_uuid") or raw.get("image") or "N/A"

    print("=" * 80)
    print(f"Pod Information: {name} ({target_pod_id})")
    print("=" * 80)
    print(f"Status:         {status} ({pod_type})")
    if created_display:
      print(f"Uptime:         {uptime_str} (created: {created_display})")
    else:
      print(f"Uptime:         {uptime_str}")
    print(f"Cost:           {cost_str}")
    print(f"Datacenter:     {location}")
    print()
    print("-- Hardware & Resources " + "-" * 56)
    print(f"GPU:            {gpu_display}")
    print(f"CPU:            {vcpu_str}")
    print(f"Memory (RAM):   {ram_str}")
    print(f"Container Disk: {disk_str}")
    print(f"Network Volume: {v_disk_str}")

    has_live_metrics = (cpu_pct is not None or mem_pct is not None or gpu_pct is not None or gpu_mem_pct is not None)
    if has_live_metrics:
      print()
      print("-- Live Utilization " + "-" * 60)
      if cpu_pct is not None:
        print(f"CPU Usage:      {cpu_pct}%")
      if mem_pct is not None:
        print(f"RAM Usage:      {mem_pct}%")
      if gpu_pct is not None:
        print(f"GPU Compute:    {gpu_pct}%")
      if gpu_mem_pct is not None:
        print(f"GPU VRAM:       {gpu_mem_pct}%")

    print()
    print("-- Network & Access " + "-" * 60)
    print(f"SSH Endpoint:   {ssh_endpoint}")
    print(f"SSH Command:    {ssh_cmd}")
    print(f"Docker Image:   {image}")
    print("=" * 80)


def cmd_stop(args):
  target_pod_id = resolve_pod_id(args)
  stop_proxy_for_pod(target_pod_id)
  provider = resolve_provider(args=args, instance_id=target_pod_id)
  print(f"Stopping pod '{target_pod_id}'...")
  try:
    if provider.name == "runpod":
      runpod.stop_pod(target_pod_id)
    else:
      provider.stop_instance(target_pod_id)
    print(f"Stop request sent for pod '{target_pod_id}'.")
  except Exception as e:
    fatal(f"Failed to stop pod: {e}", exc=e.__class__)


def cmd_terminate(args):
  target_pod_id = resolve_pod_id(args)
  stop_proxy_for_pod(target_pod_id)
  provider = resolve_provider(args=args, instance_id=target_pod_id)
  print(f"Terminating pod '{target_pod_id}'...")
  try:
    if provider.name == "runpod":
      runpod.terminate_pod(target_pod_id)
    else:
      provider.terminate_instance(target_pod_id)
    clear_last_pod_id_if_match(target_pod_id)
    print(f"Termination request sent for pod '{target_pod_id}'.")
  except Exception as e:
    fatal(f"Failed to terminate pod: {e}", exc=e.__class__)


def cmd_proxy(args):
  action = getattr(args, "proxy_action", None)

  if action == "start":
    target_pod_id = resolve_pod_id(args)
    verbose = getattr(args, "verbose", False)
    ssh_host, ssh_port = resolve_instance_ssh_endpoint(target_pod_id, args, verbose=verbose)
    priv_key = find_ssh_private_key(None, getattr(args, "ssh_private_key_path", None))
    remote_port = getattr(args, "remote_port", 8080)
    allow_write = getattr(args, "allow_write", None)
    info = start_proxy_for_pod(
      pod_id=target_pod_id,
      host=ssh_host,
      port=ssh_port,
      remote_port=remote_port,
      private_key_path=priv_key,
      ssh_config_path=getattr(args, "ssh_config", None),
      allow_write=allow_write,
      verbose=verbose
    )
    print(f"Auth proxy active for pod '{target_pod_id}':")
    print(f"  Remote tunnel: 127.0.0.1:{info.get('remote_port')}")
    print(f"  Local proxy:   127.0.0.1:{info.get('local_port')}")
    print(f"  Routes:        {', '.join(info.get('routes', [])) or 'None'}")
    if info.get("env"):
      print("  Injected environment:")
      for k, v in info["env"].items():
        print(f"    {k}={v}")
  elif action == "stop":
    target_pod_id = resolve_pod_id(args)
    stopped = stop_proxy_for_pod(pod_id=target_pod_id)
    if stopped:
      print(f"Auth proxy stopped for pod '{target_pod_id}'.")
    else:
      print(f"No active auth proxy found for pod '{target_pod_id}'.")
  elif action == "status":
    target_pod_id = resolve_pod_id(args)
    info = get_active_proxy_info(target_pod_id)
    if info:
      uptime = int(time.time() - info.get("started_at", time.time()))
      print(f"Auth proxy status for pod '{target_pod_id}': ACTIVE (PID {info.get('pid')}, uptime {uptime}s)")
      print(f"  Remote tunnel: 127.0.0.1:{info.get('remote_port')}")
      print(f"  Local proxy:   127.0.0.1:{info.get('local_port')}")
      print(f"  Routes:        {', '.join(info.get('routes', [])) or 'None'}")
      if info.get("env"):
        print("  Injected environment:")
        for k, v in info["env"].items():
          print(f"    {k}={v}")
    else:
      print(f"Auth proxy status for pod '{target_pod_id}': INACTIVE")
  elif action == "list":
    proxies = list_active_proxies()
    if not proxies:
      print("No active auth proxies.")
      return
    print(f"{'POD ID':<25} {'REMOTE PORT':<12} {'LOCAL PORT':<12} {'ROUTES':<30} {'PID':<8}")
    print("-" * 90)
    for p in proxies:
      routes_str = ", ".join(p.get("routes", []))
      print(f"{p.get('pod_id'):<25} {p.get('remote_port'):<12} {p.get('local_port'):<12} {routes_str:<30} {p.get('pid'):<8}")


def cmd_gpus(args):
  provider = resolve_provider(args=args)
  filter_pattern = getattr(args, "regex_filter", None) or getattr(args, "filter", None)
  if filter_pattern:
    try:
      re.compile(filter_pattern, re.IGNORECASE)
    except re.error as e:
      fatal(f"Invalid regex pattern '{filter_pattern}': {e}", ValueError)

  expand = getattr(args, "expand", False)
  if provider.name in ("vast", "spheron", "cloudrift", "lium"):
    if provider.name == "vast":
      prov_label = "Vast.ai"
    elif provider.name == "spheron":
      prov_label = "Spheron"
    elif provider.name == "cloudrift":
      prov_label = "CloudRift"
    else:
      prov_label = "Lium"
    print(f"Fetching available GPU types on {prov_label}...")
    try:
      gpus = provider.list_gpus(filter_pattern, expand=expand)
    except Exception as e:
      fatal(f"Failed to retrieve GPUs: {e}", exc=e.__class__)

    if not gpus:
      if filter_pattern:
        print(f"No GPUs matching pattern '{filter_pattern}' found.")
      else:
        print("No GPUs found.")
      return

    if expand:
      print(f"{'OFFER ID':<10} | {'GPU MODEL':<20} | {'GPUS':<5} | {'VRAM':<8} | {'PRICE':<10} | {'RELIABILITY':<12} | {'INET (DOWN/UP)':<18} | {'DISK':<9} | {'LOCATION'}")
      print("-" * 115)
      for o in gpus:
        offer_id = str(o.get("id") or o.get("offerId") or "N/A")
        gpu_name = str(o.get("gpu_name") or o.get("gpuType") or o.get("displayName") or "Unknown")
        num_gpus = f"{o.get('num_gpus') or o.get('gpuCount') or 1}x"
        vram = o.get("gpu_ram") or (o.get("memoryInGb") * 1024 if o.get("memoryInGb") else 0)
        vram_gb = vram // 1024 if vram > 128 else vram
        vram_str = f"{int(vram_gb)} GB" if vram_gb else "N/A"
        dph = o.get("dph_total") if o.get("dph_total") is not None else (o.get("pricePerHour") or o.get("minPrice"))
        price_str = f"${dph:.3f}/hr" if isinstance(dph, (int, float)) else "N/A"
        rel = o.get("reliability2")
        rel_str = f"{rel * 100:.1f}%" if isinstance(rel, (int, float)) else "N/A"
        down = o.get("inet_down")
        up = o.get("inet_up")
        if down is not None and up is not None:
          inet_str = f"{int(down)}/{int(up)} Mbps"
        else:
          inet_str = "N/A"
        disk = o.get("disk_space") or o.get("disk")
        disk_str = f"{int(disk)} GB" if disk else "N/A"
        loc = o.get("geolocation") or o.get("country") or o.get("region") or "N/A"
        print(f"{offer_id:<10} | {gpu_name:<20} | {num_gpus:<5} | {vram_str:<8} | {price_str:<10} | {rel_str:<12} | {inet_str:<18} | {disk_str:<9} | {loc}")
      return

    print(f"{'GPU MODEL':<25} | {'VRAM':<10} | {'MIN PRICE':<12} | {'MAX PRICE':<12} | {'AVAILABLE'}")
    print("-" * 80)
    for g in gpus:
      vram_str = f"{g['memoryInGb']} GB" if g.get("memoryInGb") else "N/A"
      min_p = f"${g['minPrice']:.2f}/hr" if g.get("minPrice") is not None else "N/A"
      max_p = f"${g['maxPrice']:.2f}/hr" if g.get("maxPrice") is not None else "N/A"
      print(f"{g['displayName']:<25} | {vram_str:<10} | {min_p:<12} | {max_p:<12} | {g['count']} offers")
    return

  print("Fetching available GPU types...")
  try:
    query = """
    query GpuTypes {

      gpuTypes {
        id
        displayName
        memoryInGb
        maxGpuCount
        securePrice
        communityPrice
      }
    }
    """
    response = run_graphql(query)
    gpus = response.get("data", {}).get("gpuTypes", [])
  except Exception:
    try:
      gpus = runpod.get_gpus()
    except Exception as e:
      fatal(f"Failed to retrieve GPUs: {e}", exc=e.__class__)

  if not gpus:
    print("No GPUs found.")
    return

  if filter_pattern:
    regex = re.compile(filter_pattern, re.IGNORECASE)
    gpus = [
        g for g in gpus
        if regex.search(g.get("id") or "") or regex.search(g.get("displayName") or "")
    ]

    if not gpus:
      print(f"No GPUs matching pattern '{filter_pattern}' found.")
      return

  if expand:
    print("Note: RunPod allocates pooled GPU capacity rather than individual marketplace offers. Use --cloud-type (SECURE/COMMUNITY) when launching.\n")

  print(f"{'GPU ID':<30} | {'DISPLAY NAME':<25} | {'VRAM (GB)':<9} | {'MAX':<3} | {'SECURE':<7} | {'COMMUNITY':<9}")
  print("-" * 98)
  for g in gpus:
    gpu_id = g.get("id") or "N/A"
    display_name = g.get("displayName") or "N/A"
    ram = g.get("memoryInGb")
    ram_str = f"{ram}" if ram is not None else "N/A"
    max_gpus = g.get("maxGpuCount")
    max_gpus_str = f"{max_gpus}" if max_gpus is not None else "N/A"
    sec_price = g.get("securePrice")
    comm_price = g.get("communityPrice")

    sec_price_str = f"${sec_price:.2f}/h" if isinstance(sec_price, (int, float)) else "N/A"
    comm_price_str = f"${comm_price:.2f}/h" if isinstance(comm_price, (int, float)) else "N/A"

    print(f"{gpu_id:<30} | {display_name:<25} | {ram_str:<9} | {max_gpus_str:<3} | {sec_price_str:<7} | {comm_price_str:<9}")


def get_pod_template(template_id):
  try:
    query = f"""
    query PodTemplate {{
      podTemplate(id: "{template_id}") {{
        id
        name
        imageName
      }}
    }}
    """
    response = run_graphql(query)
    template = response.get("data", {}).get("podTemplate")
    if template:
      return template
  except Exception:
    pass

  templates = get_pod_templates()
  for t in templates:
    if t.get("id") == template_id:
      return t

  return None


def get_pod_templates():
  try:
    query = """
    query PodTemplates {
      myself {
        podTemplates {
          id
          name
          imageName
        }
      }
    }
    """
    response = run_graphql(query)
    return response.get("data", {}).get("myself", {}).get("podTemplates", [])
  except Exception as e:
    fatal(f"Failed to retrieve templates: {e}", exc=e.__class__)


def cmd_templates(args):
  provider = resolve_provider(args=args)
  prov_label = provider.name.capitalize() if provider.name != "vast" else "Vast.ai"
  print(f"Fetching available templates and images on {prov_label}...")

  filter_pattern = getattr(args, "regex_filter", None) or getattr(args, "filter", None)
  if filter_pattern:
    try:
      re.compile(filter_pattern, re.IGNORECASE)
    except re.error as e:
      fatal(f"Invalid regex pattern '{filter_pattern}': {e}", ValueError)

  try:
    templates = provider.list_templates(filter_pattern=filter_pattern)
  except Exception as e:
    fatal(f"Failed to retrieve templates: {e}", exc=e.__class__)

  if not templates:
    if filter_pattern:
      print(f"No templates matching pattern '{filter_pattern}' found.")
    else:
      print("No templates found.")
    return

  print(f"{'TEMPLATE ID':<35} | {'TYPE':<10} | {'NAME':<32} | {'BASE IMAGE / OS'}")
  print("-" * 115)
  try:
    for t in templates:
      if hasattr(t, "id"):
        t_id = t.id
        t_type = getattr(t, "type", "template")
        name = (t.name[:32] if t.name else "N/A")
        img = getattr(t, "image_name", None) or "N/A"
      elif isinstance(t, dict):
        raw_id = t.get("id", "N/A")
        t_type = t.get("type", "template")
        t_id = format_template_identifier(t_type, raw_id) if ":" not in str(raw_id) else raw_id
        name = (t.get("name") or "N/A")[:32]
        img = t.get("imageName") or t.get("image") or "N/A"
      else:
        continue
      print(f"{t_id:<35} | {t_type:<10} | {name:<32} | {img}")
  except BrokenPipeError:
    pass


def main(args=None):
  if args is not None and args and (args[0].endswith(".py") or args[0] in ("runpod-shell", "cli.py")):
    args = args[1:]
  raw_args = list(args) if args is not None else sys.argv[1:]
  raw_args = preprocess_argv(raw_args)

  # Pass 1: Early provider resolution
  pre_parser = argparse.ArgumentParser(add_help=False)
  pre_parser.add_argument("--provider", default=None)
  pre_parser.add_argument("--offer-id", default=None)
  pre_parser.add_argument("--ssh-key-id", default=None)
  pre_parser.add_argument("--datacenter", default=None)
  pre_parser.add_argument("--executor-id", default=None)
  pre_parser.add_argument("-p", "--pod", default=None)
  pre_parser.add_argument("--api-key", default=None)
  pre_args, remaining_args = pre_parser.parse_known_args(raw_args)

  target_id = pre_args.pod
  if not target_id and remaining_args:
    for token in remaining_args:
      if not token.startswith("-"):
        prov, _ = parse_instance_identifier(token)
        if prov or (token.isdigit() and len(token) >= 4):
          target_id = token
          break

  provider = resolve_provider(args=pre_args, instance_id=target_id)

  parser = argparse.ArgumentParser(
      prog="xpu-shell",
      description="Manage AI accelerator instances (RunPod, Vast.ai, Spheron, CloudRift, Lium, etc.) with remote execution and job supervision."
  )
  parser.add_argument(
      "--api-key",
      help="API Key (can also be set via RUNPOD_API_KEY, VAST_API_KEY, SPHERON_API_KEY, CLOUDRIFT_API_KEY, or LIUM_API_KEY environment variable)"
  )

  subparser_parent = argparse.ArgumentParser(add_help=False)
  subparser_parent.add_argument(
      "--provider",
      help="Target provider ('runpod', 'vast', 'spheron', 'cloudrift', or 'lium'). Default: auto-detect or runpod.",
      default=None
  )
  subparser_parent.add_argument(
      "--api-key",
      help="API Key (can also be set via RUNPOD_API_KEY, VAST_API_KEY, SPHERON_API_KEY, CLOUDRIFT_API_KEY, or LIUM_API_KEY environment variable)",
      default=argparse.SUPPRESS
  )

  subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommands")

  # Create Command
  create_parser = subparsers.add_parser("create", parents=[subparser_parent], help="Create and launch a new instance")
  create_parser.add_argument(
      "-t",
      "--template",
      default=None,
      help="Template or image identifier (e.g. 'template:<id>' or 'image:<name>')."
  )
  create_parser.add_argument(
      "--name",
      default="persistent-worker",
      help="Name of the instance (default: %(default)s)"
  )
  create_parser.add_argument(
      "--image-name",
      "--image_name",
      dest="image_name",
      default=None,
      help="Base image name"
  )
  create_parser.add_argument(
      "--gpu-type",
      default="NVIDIA GeForce RTX 4090",
      help="GPU type ID (default: %(default)s)"
  )
  create_parser.add_argument(
      "--volume-size",
      type=int,
      default=50,
      help="Container disk volume size in GB (default: %(default)s)"
  )
  create_parser.add_argument(
      "--ssh-key-path",
      help="Path to SSH public key file. If omitted, searches default keys in ~/.ssh/"
  )
  create_parser.add_argument(
      "--requirements-path",
      help="Path to requirements.txt file (optional)"
  )
  create_parser.add_argument(
      "--pip-packages",
      "--pip_packages",
      dest="pip_packages",
      nargs="+",
      help="Extra python packages to install in the virtual environment"
  )
  create_parser.add_argument(
      "--apt-packages",
      nargs="+",
      help="Extra apt packages to install (default: screen curl htop ffmpeg git)"
  )
  create_parser.add_argument(
      "--apt-packages-file",
      "--apt_packages_file",
      dest="apt_packages_file",
      help="Path to a file containing extra apt packages to install"
  )
  create_parser.add_argument(
      "--env",
      nargs="+",
      action=ParseEnv,
      default={},
      help="Extra environment variables to set in the container (e.g. KEY=VALUE KEY2=VALUE2)"
  )
  create_parser.add_argument(
      "--env-file",
      "--env_file",
      dest="env_file",
      help="Path to a .env file containing environment variables to set in the container"
  )
  create_parser.add_argument(
      "--gpu-count",
      type=int,
      default=1,
      help="Number of GPUs to allocate (default: %(default)s)"
  )
  create_parser.add_argument(
      "--container-disk-size",
      type=int,
      default=30,
      help="Container local disk size in GB (default: %(default)s)"
  )
  create_parser.add_argument(
      "--volume-mount-path",
      "--volume_mount_path",
      default="/workspace",
      dest="volume_mount_path",
      help="Path inside the container where the volume is mounted (default: %(default)s)"
  )
  create_parser.add_argument(
      "--run-script",
      dest="run_script",
      help="Path to a local script to execute on the pod via SSH once initialized"
  )
  create_parser.add_argument(
      "--script-args",
      dest="script_args",
      default="",
      help="String arguments to pass to the script"
  )
  create_parser.add_argument(
      "-d",
      "--detach",
      dest="detach",
      action="store_true",
      help="Run the script in background on the pod without waiting"
  )
  create_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  create_parser.add_argument(
      "--no-wait-for-setup",
      dest="no_wait_for_setup",
      action="store_true",
      help="Do not wait for container disk setup to finish before executing script"
  )
  create_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )
  create_parser.add_argument(
      "--ssh-timeout",
      dest="ssh_timeout",
      type=int,
      default=180,
      help="Max seconds to wait for SSH and setup readiness (default: %(default)s)"
  )
  create_parser.add_argument(
      "--auth-proxy",
      action="store_true",
      default=False,
      help="Automatically start a local auth-injecting proxy and SSH tunnel for the new instance"
  )
  create_parser.add_argument(
      "--proxy-remote-port",
      type=int,
      default=8080,
      help="Remote port for the reverse tunnel when --auth-proxy is used (default: 8080)"
  )
  create_parser.add_argument(
      "--proxy-allow-write",
      default=None,
      help="Semicolon-separated path/bucket patterns permitted for write operations through the proxy"
  )

  # Register provider-specific create arguments and subcommands
  if isinstance(provider, BaseProvider):
    provider.register_create_args(create_parser)
    provider.register_commands(subparsers)
  else:
    prov_name = getattr(provider, "name", "runpod")
    real_prov = get_provider(prov_name)
    real_prov.register_create_args(create_parser)
    real_prov.register_commands(subparsers)
    if hasattr(provider, "register_create_args"):
      try:
        provider.register_create_args(create_parser)
      except Exception:
        pass
    if hasattr(provider, "register_commands"):
      try:
        provider.register_commands(subparsers)
      except Exception:
        pass

  # Exec Command
  exec_parser = subparsers.add_parser("exec", parents=[subparser_parent], help="Execute a local script on an active XPU instance via SSH")
  exec_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the target pod (defaults to last created pod)"
  )
  exec_parser.add_argument(
      "script_or_pod",
      help="Path to the local script to run, or pod ID if script follows"
  )
  exec_parser.add_argument(
      "maybe_script",
      nargs="?",
      default=None,
      help="Path to the local script to run (if pod ID was specified first)"
  )
  exec_parser.add_argument(
      "--script-args",
      dest="script_args",
      default="",
      help="String arguments to pass to the script"
  )
  exec_parser.add_argument(
      "-d",
      "--detach",
      dest="detach",
      action="store_true",
      help="Run the script in background on the pod without waiting"
  )
  exec_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  exec_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )
  exec_parser.add_argument(
      "--no-wait-for-setup",
      dest="no_wait_for_setup",
      action="store_true",
      help="Do not wait for container disk setup to finish before executing script"
  )
  exec_parser.add_argument(
      "--ssh-timeout",
      dest="ssh_timeout",
      type=int,
      default=180,
      help="Max seconds to wait for SSH and setup readiness (default: %(default)s)"
  )
  exec_parser.add_argument(
      "-e",
      "--env",
      dest="env",
      action=ParseEnv,
      default={},
      help="Environment variable to set for the remote script (KEY=VALUE or KEY to extract from local environment). Can be specified multiple times."
  )
  exec_parser.add_argument(
      "--env-file",
      "--env_file",
      dest="env_files",
      action="append",
      default=[],
      help="Path to a file containing environment variables (KEY=VALUE format). Can be specified multiple times."
  )
  exec_parser.add_argument(
      "-v",
      "--verbose",
      dest="verbose",
      action="store_true",
      default=False,
      help="Show detailed job information and connection progress"
  )

  # Run Command
  run_parser = subparsers.add_parser("run", parents=[subparser_parent], help="Run a command directly (binary + args) on an active XPU instance via SSH")
  run_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the target pod (defaults to last created pod)"
  )
  run_parser.add_argument(
      "-d",
      "--detach",
      dest="detach",
      action="store_true",
      help="Run the command in background on the pod without waiting"
  )
  run_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  run_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )
  run_parser.add_argument(
      "--no-wait-for-setup",
      dest="no_wait_for_setup",
      action="store_true",
      help="Do not wait for container disk setup to finish before executing command"
  )
  run_parser.add_argument(
      "--ssh-timeout",
      dest="ssh_timeout",
      type=int,
      default=180,
      help="Max seconds to wait for SSH and setup readiness (default: %(default)s)"
  )
  run_parser.add_argument(
      "-e",
      "--env",
      dest="env",
      action=ParseEnv,
      default={},
      help="Environment variable to set for the remote command (KEY=VALUE or KEY to extract from local environment). Can be specified multiple times."
  )
  run_parser.add_argument(
      "--env-file",
      "--env_file",
      dest="env_files",
      action="append",
      default=[],
      help="Path to a file containing environment variables (KEY=VALUE format). Can be specified multiple times."
  )
  run_parser.add_argument(
      "-s",
      "--shell",
      dest="use_shell",
      action="store_true",
      help="Execute command within a remote shell (enables wildcards, pipes, redirects)"
  )
  run_parser.add_argument(
      "-v",
      "--verbose",
      dest="verbose",
      action="store_true",
      default=False,
      help="Show detailed job information and connection progress"
  )
  run_parser.add_argument(
      "cmd",
      nargs=argparse.REMAINDER,
      help="Command line to execute directly on the remote pod (binary + args)"
  )

  # Cp Command
  cp_parser = subparsers.add_parser("cp", parents=[subparser_parent], help="Copy files or directories between a local path and an XPU instance via SCP or Rsync")
  cp_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the target pod (defaults to last created pod)"
  )
  cp_parser.add_argument(
      "-r",
      "-R",
      "--recursive",
      dest="recursive",
      action="store_true",
      help="Recursively copy entire directories"
  )
  cp_parser.add_argument(
      "-P",
      "--preserve",
      dest="preserve",
      action="store_true",
      help="Preserves modification times, access times, and modes from original file"
  )
  cp_parser.add_argument(
      "-q",
      "--quiet",
      dest="quiet",
      action="store_true",
      help="Quiet mode: disables progress meter and non-fatal messages"
  )
  cp_parser.add_argument(
      "--rsync",
      "--sync",
      dest="rsync",
      action="store_true",
      help="Use rsync over SSH instead of SCP (faster incremental directory sync and compression)"
  )
  cp_parser.add_argument(
      "-a",
      "--archive",
      dest="archive",
      action="store_true",
      default=False,
      help="Archive mode (rsync mode, equals -rlptgoD)"
  )
  cp_parser.add_argument(
      "-z",
      "--compress",
      dest="compress",
      action="store_true",
      default=True,
      help="Compress file data during transfer (rsync mode, default: True)"
  )
  cp_parser.add_argument(
      "--no-compress",
      dest="compress",
      action="store_false",
      help="Disable compression during transfer (rsync mode)"
  )
  cp_parser.add_argument(
      "--delete",
      dest="delete",
      action="store_true",
      help="Delete extraneous files from destination (rsync mode)"
  )
  cp_parser.add_argument(
      "--exclude",
      dest="exclude",
      action="append",
      default=[],
      help="Exclude files matching PATTERN (rsync mode, can be specified multiple times)"
  )
  cp_parser.add_argument(
      "-n",
      "--dry-run",
      dest="dry_run",
      action="store_true",
      help="Perform a trial run with no changes made (rsync mode)"
  )
  cp_parser.add_argument(
      "-v",
      "--verbose",
      dest="verbose",
      action="store_true",
      help="Verbose output (rsync mode)"
  )
  cp_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  cp_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )
  cp_parser.add_argument(
      "paths",
      nargs="+",
      help="Source and destination paths (e.g. local_file.txt :/workspace/ or :/workspace/data.csv ./)"
  )

  # Rsync Command
  rsync_parser = subparsers.add_parser("rsync", parents=[subparser_parent], help="Synchronize files or directories between a local path and an XPU instance via rsync over SSH")
  rsync_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the target pod (defaults to last created pod)"
  )
  rsync_parser.add_argument(
      "-a",
      "--archive",
      dest="archive",
      action="store_true",
      default=True,
      help="Archive mode (default: True, equals -rlptgoD)"
  )
  rsync_parser.add_argument(
      "--no-archive",
      dest="archive",
      action="store_false",
      help="Disable archive mode"
  )
  rsync_parser.add_argument(
      "-r",
      "-R",
      "--recursive",
      dest="recursive",
      action="store_true",
      help="Recursively sync directories"
  )
  rsync_parser.add_argument(
      "-P",
      "--preserve",
      dest="preserve",
      action="store_true",
      help="Preserve modification times, access times, and modes"
  )
  rsync_parser.add_argument(
      "-z",
      "--compress",
      dest="compress",
      action="store_true",
      default=True,
      help="Compress file data during transfer (default: True)"
  )
  rsync_parser.add_argument(
      "--no-compress",
      dest="compress",
      action="store_false",
      help="Disable compression during transfer"
  )
  rsync_parser.add_argument(
      "--delete",
      dest="delete",
      action="store_true",
      help="Delete extraneous files from destination"
  )
  rsync_parser.add_argument(
      "--exclude",
      dest="exclude",
      action="append",
      default=[],
      help="Exclude files matching PATTERN (can be specified multiple times)"
  )
  rsync_parser.add_argument(
      "-n",
      "--dry-run",
      dest="dry_run",
      action="store_true",
      help="Perform a trial run with no changes made"
  )
  rsync_parser.add_argument(
      "-q",
      "--quiet",
      dest="quiet",
      action="store_true",
      help="Quiet mode: disables progress meter and messages"
  )
  rsync_parser.add_argument(
      "-v",
      "--verbose",
      dest="verbose",
      action="store_true",
      help="Verbose output"
  )
  rsync_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  rsync_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )
  rsync_parser.add_argument(
      "paths",
      nargs="+",
      help="Source and destination paths (e.g. ./src/ :/workspace/src/ or :/workspace/data/ ./data/)"
  )

  # Ps Command
  ps_parser = subparsers.add_parser("ps", parents=[subparser_parent], help="List remote processes and managed jobs on an XPU instance")
  ps_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the pod (defaults to last created pod)"
  )
  ps_parser.add_argument(
      "pod_id",
      nargs="?",
      default=None,
      help="The ID of the pod (optional, defaults to last created pod)"
  )
  ps_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  ps_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )

  # Logs Command
  logs_parser = subparsers.add_parser("logs", parents=[subparser_parent], help="View remote logs of managed jobs on an XPU instance")
  logs_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the pod (defaults to last created pod)"
  )
  logs_parser.add_argument(
      "-j",
      "--job",
      dest="job",
      default=None,
      help="The Job ID or PID (defaults to last job ID)"
  )
  logs_parser.add_argument(
      "arg1",
      nargs="?",
      default=None,
      help="The ID of the pod, or Job ID/PID if pod is already specified or defaulted"
  )
  logs_parser.add_argument(
      "arg2",
      nargs="?",
      default=None,
      help="The Job ID or PID"
  )
  logs_parser.add_argument(
      "-n",
      "--tail",
      dest="tail",
      type=int,
      default=None,
      help="Number of lines to display from end of log"
  )
  logs_parser.add_argument(
      "-f",
      "--follow",
      dest="follow",
      action="store_true",
      help="Follow log output in real-time"
  )
  logs_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  logs_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )
  logs_parser.add_argument(
      "-v",
      "--verbose",
      dest="verbose",
      action="store_true",
      default=False,
      help="Show detailed job information and connection progress"
  )

  # Kill Command
  kill_parser = subparsers.add_parser("kill", parents=[subparser_parent], help="Kill a remote job or process on an XPU instance")
  kill_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the pod (defaults to last created pod)"
  )
  kill_parser.add_argument(
      "-j",
      "--job",
      dest="job",
      default=None,
      help="The Job ID or PID to kill (defaults to last job ID)"
  )
  kill_parser.add_argument(
      "target_or_pod",
      nargs="?",
      default=None,
      help="The Job ID or PID to kill, or pod ID if target follows (optional, defaults to last job ID)"
  )
  kill_parser.add_argument(
      "maybe_target",
      nargs="?",
      default=None,
      help="The Job ID or PID to kill (if pod ID specified first)"
  )
  kill_parser.add_argument(
      "-s",
      "--signal",
      dest="signal",
      default="SIGTERM",
      help="Signal to send (e.g. SIGTERM, SIGKILL; default: %(default)s)"
  )
  kill_parser.add_argument(
      "-t",
      "--timeout",
      dest="timeout",
      type=float,
      default=30.0,
      help="Grace period in seconds before escalating SIGTERM to SIGKILL (default: %(default)s)"
  )
  kill_parser.add_argument(
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      help="Path to private SSH key (auto-detected if omitted)"
  )
  kill_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file (e.g. /dev/null), or RUNPOD_SSH_CONFIG env var"
  )

  # List Command
  subparsers.add_parser("list", parents=[subparser_parent], help="List all your XPU instances")

  # Info Command
  info_parser = subparsers.add_parser("info", parents=[subparser_parent], help="Show detailed information about an XPU instance")
  info_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the pod (defaults to last created pod)"
  )
  info_parser.add_argument(
      "pod_id",
      nargs="?",
      default=None,
      help="The ID of the pod (optional, defaults to last created pod)"
  )
  info_parser.add_argument(
      "--json",
      action="store_true",
      dest="json_output",
      default=False,
      help="Output raw pod details in JSON format"
  )

  # Stop Command
  stop_parser = subparsers.add_parser("stop", parents=[subparser_parent], help="Stop a running XPU instance")
  stop_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the pod to stop (defaults to last created pod)"
  )
  stop_parser.add_argument(
      "pod_id",
      nargs="?",
      default=None,
      help="The ID of the pod to stop (optional, defaults to last created pod)"
  )

  # Terminate Command
  terminate_parser = subparsers.add_parser("terminate", parents=[subparser_parent], help="Terminate (delete) an XPU instance")
  terminate_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the pod to terminate (defaults to last created pod)"
  )
  terminate_parser.add_argument(
      "pod_id",
      nargs="?",
      default=None,
      help="The ID of the pod to terminate (optional, defaults to last created pod)"
  )

  # Gpus Command
  gpus_parser = subparsers.add_parser("gpus", parents=[subparser_parent], help="List all available GPU types and details on XPU")
  gpus_parser.add_argument(
      "filter",
      nargs="?",
      default=None,
      help="Optional regex pattern to filter GPUs by ID or Display Name (case-insensitive)"
  )
  gpus_parser.add_argument(
      "-r",
      "--regex",
      "--filter",
      dest="regex_filter",
      default=None,
      help="Optional regex pattern to filter GPUs by ID or Display Name (case-insensitive)"
  )
  gpus_parser.add_argument(
      "-l",
      "--expand",
      action="store_true",
      dest="expand",
      default=False,
      help="Expand list to show individual marketplace offerings (e.g. for Vast.ai)"
  )

  # Templates Command
  templates_parser = subparsers.add_parser("templates", parents=[subparser_parent], help="List available templates and images on XPU")
  templates_parser.add_argument(
      "filter",
      nargs="?",
      default=None,
      help="Optional regex pattern to filter templates by ID, Name, or Image (case-insensitive)"
  )
  templates_parser.add_argument(
      "-r",
      "--regex",
      "--filter",
      dest="regex_filter",
      default=None,
      help="Optional regex pattern to filter templates by ID, Name, or Image (case-insensitive)"
  )

  # Proxy Command
  proxy_parser = subparsers.add_parser("proxy", parents=[subparser_parent], help="Manage auth-injecting proxies and SSH reverse tunnels for XPU instances")
  proxy_parser.add_argument(
      "proxy_action",
      choices=["start", "stop", "status", "list"],
      help="Action to perform: start, stop, status, or list"
  )
  proxy_parser.add_argument(
      "-p",
      "--pod",
      dest="pod",
      default=None,
      help="The ID of the target pod (defaults to last created pod)"
  )
  proxy_parser.add_argument(
      "--remote-port",
      type=int,
      default=8080,
      help="Port on the remote instance for the reverse tunnel (default: 8080)"
  )
  proxy_parser.add_argument(
      "--allow-write",
      default=None,
      help="Semicolon-separated path/bucket patterns permitted for write operations (e.g. 's3://my-bucket/checkpoints/*')"
  )
  proxy_parser.add_argument(
      "-i",
      "--ssh-private-key-path",
      dest="ssh_private_key_path",
      default=None,
      help="Path to SSH private key file"
  )
  proxy_parser.add_argument(
      "--ssh-config",
      dest="ssh_config",
      default=os.environ.get("RUNPOD_SSH_CONFIG"),
      help="Path to custom SSH config file"
  )
  proxy_parser.add_argument(
      "-v",
      "--verbose",
      action="store_true",
      dest="verbose",
      default=False,
      help="Show detailed proxy and connection progress"
  )

  args = parser.parse_args(raw_args)

  # Set API Key and resolve provider
  target_id = getattr(args, "pod", None) or getattr(args, "pod_id", None) or target_id
  provider = resolve_provider(args=args, instance_id=target_id)

  if args.api_key:
    if provider.name == "vast":
      os.environ["VAST_API_KEY"] = args.api_key
    elif provider.name == "spheron":
      os.environ["SPHERON_API_KEY"] = args.api_key
    elif provider.name == "cloudrift":
      os.environ["CLOUDRIFT_API_KEY"] = args.api_key
    elif provider.name == "lium":
      os.environ["LIUM_API_KEY"] = args.api_key
    else:
      runpod.api_key = args.api_key
  elif not runpod.api_key:
    runpod.api_key = os.environ.get("RUNPOD_API_KEY")

  if provider.name == "runpod" and not runpod.api_key:
    parser.error(
        "RunPod API key must be provided via --api-key or the RUNPOD_API_KEY environment variable."
    )


  if args.command == "create":
    cmd_create(args)
  elif args.command == "run":
    cmd_run(args)
  elif args.command in ("cp", "rsync"):
    cmd_cp(args)
  elif args.command == "list":
    cmd_list(args)
  elif args.command == "info":
    cmd_info(args)
  elif args.command == "stop":
    cmd_stop(args)
  elif args.command == "terminate":
    cmd_terminate(args)
  elif args.command == "gpus":
    cmd_gpus(args)
  elif args.command == "templates":
    cmd_templates(args)
  elif args.command == "exec":
    cmd_exec(args)
  elif args.command == "ps":
    cmd_ps(args)
  elif args.command == "logs":
    cmd_logs(args)
  elif args.command == "kill":
    cmd_kill(args)
  elif args.command == "proxy":
    cmd_proxy(args)


if __name__ == "__main__":
  main()
