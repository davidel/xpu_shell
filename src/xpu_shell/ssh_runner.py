import base64
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid


RUNNER_LOCAL_PATH = Path(__file__).parent / "runner.py"
REMOTE_RUNNER_PATH = "/tmp/.runpod_runner.py"
_ENSURED_RUNNER_HOSTS = set()


def find_ssh_private_key(public_key_path=None, explicit_private_key_path=None):
  if explicit_private_key_path:
    priv_path = Path(explicit_private_key_path).expanduser()
    if not priv_path.exists():
      raise FileNotFoundError(f"SSH private key not found at {priv_path}")
    return priv_path

  if public_key_path:
    pub_path = Path(public_key_path).expanduser()
    if pub_path.name.endswith(".pub"):
      candidate = pub_path.with_name(pub_path.name[:-4])
      if candidate.exists():
        return candidate
    if pub_path.exists():
      return pub_path

  ssh_dir = Path.home() / ".ssh"
  if ssh_dir.exists():
    for name in ["id_rsa", "id_ed25519", "id_ecdsa", "id_dsa"]:
      candidate = ssh_dir / name
      if candidate.exists():
        return candidate

  return None


def resolve_ssh_config(explicit_config=None):
  if explicit_config is not None:
    if str(explicit_config).lower() in ("none", "system", "default", ""):
      return None
    return str(explicit_config)

  env_config = os.environ.get("RUNPOD_SSH_CONFIG")
  if env_config is not None:
    if env_config.lower() in ("none", "system", "default", ""):
      return None
    return env_config

  user_ssh_config = Path.home() / ".ssh" / "config"
  if user_ssh_config.is_file():
    return str(user_ssh_config)

  conf_d = Path("/etc/ssh/ssh_config.d")
  if conf_d.is_dir():
    try:
      for f in conf_d.glob("*.conf"):
        if f.stat().st_uid not in (0, os.getuid()):
          return "/dev/null"
    except Exception:
      pass

  return None


def build_ssh_cmd(host, port, remote_command=None, private_key_path=None, tty=False, ssh_config_path=None):
  cmd = ["ssh", "-p", str(port)]
  cfg = resolve_ssh_config(ssh_config_path)
  if cfg:
    cmd.extend(["-F", str(cfg)])
  cmd.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"])
  if private_key_path:
    cmd.extend(["-i", str(private_key_path)])
  if tty:
    cmd.append("-t")
  cmd.append(f"root@{host}")
  if remote_command:
    cmd.append(remote_command)
  return cmd


def build_scp_cmd(local_path, remote_path, host, port, private_key_path=None, ssh_config_path=None):
  cmd = ["scp", "-P", str(port)]
  cfg = resolve_ssh_config(ssh_config_path)
  if cfg:
    cmd.extend(["-F", str(cfg)])
  cmd.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"])
  if private_key_path:
    cmd.extend(["-i", str(private_key_path)])
  cmd.extend([str(local_path), f"root@{host}:{remote_path}"])
  return cmd


def build_cp_cmd(
    sources,
    dest,
    port,
    recursive=False,
    preserve=False,
    quiet=False,
    private_key_path=None,
    ssh_config_path=None
):
  cmd = ["scp", "-P", str(port)]
  if recursive:
    cmd.append("-r")
  if preserve:
    cmd.append("-p")
  if quiet:
    cmd.append("-q")
  cfg = resolve_ssh_config(ssh_config_path)
  if cfg:
    cmd.extend(["-F", str(cfg)])
  cmd.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"])
  if private_key_path:
    cmd.extend(["-i", str(private_key_path)])
  for s in sources:
    cmd.append(str(s))
  cmd.append(str(dest))
  return cmd


def ensure_remote_runner(host, port, private_key_path=None, ssh_config_path=None):
  cache_key = (str(host), int(port))
  if cache_key in _ENSURED_RUNNER_HOSTS:
    return REMOTE_RUNNER_PATH

  if not RUNNER_LOCAL_PATH.exists():
    raise FileNotFoundError(f"Local runner script not found at {RUNNER_LOCAL_PATH}")

  scp_cmd = build_scp_cmd(
      RUNNER_LOCAL_PATH,
      REMOTE_RUNNER_PATH,
      host,
      port,
      private_key_path=private_key_path,
      ssh_config_path=ssh_config_path
  )
  res = subprocess.run(scp_cmd, capture_output=True, text=True)
  if res.returncode != 0:
    raise RuntimeError(f"Failed to upload runner to pod via SCP: {res.stderr.strip()}")

  _ENSURED_RUNNER_HOSTS.add(cache_key)
  return REMOTE_RUNNER_PATH


def wait_for_ssh(host, port, private_key_path=None, timeout=180, interval=3, ssh_config_path=None, verbose=False):
  if verbose:
    print(f"Waiting for SSH daemon at {host}:{port} to become available...", file=sys.stderr)
  start_time = time.time()
  while time.time() - start_time < timeout:
    cmd = build_ssh_cmd(host, port, "true", private_key_path=private_key_path, ssh_config_path=ssh_config_path)
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
      if res.returncode == 0:
        if verbose:
          print("SSH connection established successfully.", file=sys.stderr)
        return True
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
      pass
    time.sleep(interval)
  raise TimeoutError(f"Timed out waiting for SSH daemon at {host}:{port} after {timeout} seconds.")


def wait_for_setup(host, port, private_key_path=None, timeout=300, interval=5, ssh_config_path=None, verbose=False):
  if verbose:
    print("Waiting for container disk and environment setup to complete...", file=sys.stderr)
  check_cmd = "test -f /workspace/.setup_complete || test -f /tmp/.setup_complete"
  start_time = time.time()
  while time.time() - start_time < timeout:
    cmd = build_ssh_cmd(host, port, check_cmd, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
    try:
      res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
      if res.returncode == 0:
        if verbose:
          print("Container environment setup is complete.", file=sys.stderr)
        return True
    except (subprocess.TimeoutExpired, subprocess.SubprocessError):
      pass
    time.sleep(interval)
  raise TimeoutError("Timed out waiting for container setup sentinel file.")


def execute_remote_script(
    host,
    port,
    script_path,
    script_args="",
    detach=False,
    private_key_path=None,
    wait_for_setup_flag=True,
    ssh_timeout=180,
    ssh_config_path=None,
    extra_env=None,
    verbose=False
):
  local_path = Path(script_path).expanduser()
  if not local_path.exists():
    raise FileNotFoundError(f"Local script not found at {local_path}")

  wait_for_ssh(host, port, private_key_path=private_key_path, timeout=ssh_timeout, ssh_config_path=ssh_config_path, verbose=verbose)

  if wait_for_setup_flag:
    wait_for_setup(host, port, private_key_path=private_key_path, timeout=ssh_timeout, ssh_config_path=ssh_config_path, verbose=verbose)

  ensure_remote_runner(host, port, private_key_path=private_key_path, ssh_config_path=ssh_config_path)

  upload_prefix = f"upload_{int(time.time())}_{uuid.uuid4().hex[:6]}"
  remote_script_path = f"/tmp/{upload_prefix}_{local_path.name}"

  if verbose:
    print(f"Uploading script '{local_path.name}' to remote pod...", file=sys.stderr)
  scp_cmd = build_scp_cmd(local_path, remote_script_path, host, port, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  upload_res = subprocess.run(scp_cmd, capture_output=True, text=True)
  if upload_res.returncode != 0:
    raise RuntimeError(f"Failed to upload script via SCP: {upload_res.stderr.strip()}")

  script_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', local_path.name) or "job"
  spawn_cmd_parts = [
      f"python3 {REMOTE_RUNNER_PATH} spawn",
      f"--script {shlex.quote(str(remote_script_path))}",
      f"--name {shlex.quote(str(script_name))}"
  ]
  if script_args:
    spawn_cmd_parts.append(f"--args {shlex.quote(str(script_args))}")
  if verbose:
    spawn_cmd_parts.append("--verbose")

  remote_spawn_cmd = " ".join(spawn_cmd_parts)
  if extra_env:
    env_json = json.dumps(extra_env)
    env_b64 = base64.b64encode(env_json.encode("utf-8")).decode("ascii")
    remote_spawn_cmd = f'export RUNPOD_JOB_ENV="{env_b64}"\n{remote_spawn_cmd}'

  launch_cmd = build_ssh_cmd(host, port, remote_spawn_cmd, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  launch_res = subprocess.run(launch_cmd, capture_output=True, text=True)
  if launch_res.returncode != 0:
    raise RuntimeError(f"Failed to launch script on pod: {launch_res.stderr.strip()}")

  pid = None
  log_file = None
  job_id = None
  for line in launch_res.stdout.splitlines():
    if line.startswith("PID:"):
      pid = line.split("PID:", 1)[1].strip()
    elif line.startswith("LOG_FILE:"):
      log_file = line.split("LOG_FILE:", 1)[1].strip()
    elif line.startswith("JOB_ID:"):
      job_id = line.split("JOB_ID:", 1)[1].strip()

  if not job_id:
    job_id = "unknown"

  if verbose:
    print(f"\nRemote job registered:", file=sys.stderr)
    print(f"  Job ID:   {job_id}", file=sys.stderr)
    print(f"  PID:      {pid}", file=sys.stderr)
    print(f"  Log file: {log_file}", file=sys.stderr)

  if detach:
    if verbose:
      print(f"\nScript is running in background.", file=sys.stderr)
      print(f"To monitor logs: runpod-shell logs [--pod <pod-id>] {job_id} -f", file=sys.stderr)
    else:
      print(job_id)
    return {"job_id": job_id, "pid": pid, "log_file": log_file, "exit_code": 0}

  # Foreground mode: stream logs until completion
  if verbose:
    print("\nStreaming remote logs (Ctrl+C to detach without stopping job)...", file=sys.stderr)
  tail_part = f"tail -n +1 -s 0.2 --pid={pid} -f '{log_file}' 2>/dev/null || tail -n +1 -f '{log_file}'"
  remote_cmd = f"bash -c '{tail_part} & TPID=$!; trap \"kill -9 $TPID 2>/dev/null\" EXIT INT TERM HUP; wait $TPID'"
  tail_cmd = build_ssh_cmd(
      host,
      port,
      remote_cmd,
      private_key_path=private_key_path,
      ssh_config_path=ssh_config_path
  )

  try:
    subprocess.run(tail_cmd)
  except KeyboardInterrupt:
    if verbose:
      print(f"\nDetached from remote process {pid}. Job continues running in background.", file=sys.stderr)
      print(f"To re-attach: runpod-shell logs [--pod <pod-id>] {job_id} -f", file=sys.stderr)
    return {"job_id": job_id, "pid": pid, "log_file": log_file, "exit_code": 0}

  # Check final exit code
  exit_code_cmd = build_ssh_cmd(
      host,
      port,
      f"cat /workspace/.runpod_jobs/{job_id}/exit_code 2>/dev/null || cat /tmp/.runpod_jobs/{job_id}/exit_code 2>/dev/null || echo unknown",
      private_key_path=private_key_path,
      ssh_config_path=ssh_config_path
  )
  code_res = subprocess.run(exit_code_cmd, capture_output=True, text=True)
  exit_code_str = code_res.stdout.strip()
  exit_code = int(exit_code_str) if exit_code_str.isdigit() else 0

  if verbose:
    print(f"\nRemote job completed with exit code: {exit_code}", file=sys.stderr)
  return {"job_id": job_id, "pid": pid, "log_file": log_file, "exit_code": exit_code}


def execute_remote_command(
    host,
    port,
    command_args,
    detach=False,
    private_key_path=None,
    wait_for_setup_flag=True,
    ssh_timeout=180,
    ssh_config_path=None,
    extra_env=None,
    use_shell=False,
    verbose=False
):
  if use_shell:
    if isinstance(command_args, (list, tuple)):
      if len(command_args) == 1:
        cmd_str = str(command_args[0])
      else:
        cmd_str = " ".join(str(a) for a in command_args)
    else:
      cmd_str = str(command_args)
    binary_name = "shell"
  else:
    if isinstance(command_args, (list, tuple)):
      cmd_str = " ".join(shlex.quote(str(a)) for a in command_args)
      first_token = command_args[0] if command_args else "cmd"
      binary_name = Path(str(first_token)).name
    else:
      cmd_str = str(command_args)
      tokens = shlex.split(cmd_str)
      binary_name = Path(tokens[0]).name if tokens else "cmd"

  binary_name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', binary_name) or "cmd"

  wait_for_ssh(host, port, private_key_path=private_key_path, timeout=ssh_timeout, ssh_config_path=ssh_config_path, verbose=verbose)

  if wait_for_setup_flag:
    wait_for_setup(host, port, private_key_path=private_key_path, timeout=ssh_timeout, ssh_config_path=ssh_config_path, verbose=verbose)

  ensure_remote_runner(host, port, private_key_path=private_key_path, ssh_config_path=ssh_config_path)

  spawn_cmd_parts = [
      f"python3 {REMOTE_RUNNER_PATH} spawn",
      f"--cmd {shlex.quote(str(cmd_str))}",
      f"--name {shlex.quote(str(binary_name))}"
  ]
  if use_shell:
    spawn_cmd_parts.append("--shell")
  if verbose:
    spawn_cmd_parts.append("--verbose")

  remote_spawn_cmd = " ".join(spawn_cmd_parts)
  if extra_env:
    env_json = json.dumps(extra_env)
    env_b64 = base64.b64encode(env_json.encode("utf-8")).decode("ascii")
    remote_spawn_cmd = f'export RUNPOD_JOB_ENV="{env_b64}"\n{remote_spawn_cmd}'

  launch_cmd = build_ssh_cmd(host, port, remote_spawn_cmd, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  launch_res = subprocess.run(launch_cmd, capture_output=True, text=True)
  if launch_res.returncode != 0:
    raise RuntimeError(f"Failed to launch command on pod: {launch_res.stderr.strip()}")

  pid = None
  log_file = None
  job_id = None
  for line in launch_res.stdout.splitlines():
    if line.startswith("PID:"):
      pid = line.split("PID:", 1)[1].strip()
    elif line.startswith("LOG_FILE:"):
      log_file = line.split("LOG_FILE:", 1)[1].strip()
    elif line.startswith("JOB_ID:"):
      job_id = line.split("JOB_ID:", 1)[1].strip()

  if not job_id:
    job_id = "unknown"

  if verbose:
    print(f"\nRemote job registered:", file=sys.stderr)
    print(f"  Job ID:   {job_id}", file=sys.stderr)
    print(f"  PID:      {pid}", file=sys.stderr)
    print(f"  Log file: {log_file}", file=sys.stderr)

  if detach:
    if verbose:
      print(f"\nCommand is running in background.", file=sys.stderr)
      print(f"To monitor logs: runpod-shell logs [-p <pod-id>] [-j <job-id>] -f", file=sys.stderr)
    else:
      print(job_id)
    return {"job_id": job_id, "pid": pid, "log_file": log_file, "exit_code": 0}

  # Foreground mode: stream logs until completion
  if verbose:
    print("\nStreaming remote logs (Ctrl+C to detach without stopping job)...", file=sys.stderr)
  tail_part = f"tail -n +1 -s 0.2 --pid={pid} -f '{log_file}' 2>/dev/null || tail -n +1 -f '{log_file}'"
  remote_cmd = f"bash -c '{tail_part} & TPID=$!; trap \"kill -9 $TPID 2>/dev/null\" EXIT INT TERM HUP; wait $TPID'"
  tail_cmd = build_ssh_cmd(
      host,
      port,
      remote_cmd,
      private_key_path=private_key_path,
      ssh_config_path=ssh_config_path
  )

  try:
    subprocess.run(tail_cmd)
  except KeyboardInterrupt:
    if verbose:
      print(f"\nDetached from remote process {pid}. Job continues running in background.", file=sys.stderr)
      print(f"To re-attach: runpod-shell logs [-p <pod-id>] [-j <job-id>] -f", file=sys.stderr)
    return {"job_id": job_id, "pid": pid, "log_file": log_file, "exit_code": 0}

  # Check final exit code
  exit_code_cmd = build_ssh_cmd(
      host,
      port,
      f"cat /workspace/.runpod_jobs/{job_id}/exit_code 2>/dev/null || cat /tmp/.runpod_jobs/{job_id}/exit_code 2>/dev/null || echo unknown",
      private_key_path=private_key_path,
      ssh_config_path=ssh_config_path
  )
  code_res = subprocess.run(exit_code_cmd, capture_output=True, text=True)
  exit_code_str = code_res.stdout.strip()
  exit_code = int(exit_code_str) if exit_code_str.isdigit() else 0

  if verbose:
    print(f"\nRemote job completed with exit code: {exit_code}", file=sys.stderr)
  return {"job_id": job_id, "pid": pid, "log_file": log_file, "exit_code": exit_code}


def list_remote_jobs(host, port, private_key_path=None, ssh_config_path=None):
  ensure_remote_runner(host, port, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  cmd = build_ssh_cmd(
      host,
      port,
      f"python3 {REMOTE_RUNNER_PATH} list",
      private_key_path=private_key_path,
      ssh_config_path=ssh_config_path
  )
  res = subprocess.run(cmd, capture_output=True, text=True)
  if res.returncode != 0:
    raise RuntimeError(f"Failed to list remote jobs: {res.stderr.strip()}")

  try:
    return json.loads(res.stdout.strip())
  except json.JSONDecodeError:
    return []


def view_remote_logs(host, port, job_id=None, tail_lines=None, follow=False, private_key_path=None, ssh_config_path=None, verbose=False):
  jobs = list_remote_jobs(host, port, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  target_job = None

  if job_id:
    for j in jobs:
      if j.get("job_id") in (str(job_id), f"job-{job_id}") or str(j.get("pid")) == str(job_id):
        target_job = j
        break
    if not target_job:
      raise ValueError(f"Job '{job_id}' not found on pod.")
  else:
    if not jobs:
      if verbose:
        print("No jobs found on pod.", file=sys.stderr)
      return
    if len(jobs) == 1:
      target_job = jobs[0]
      if verbose:
        print(f"Selecting only active job: {target_job['job_id']}", file=sys.stderr)
    else:
      target_job = jobs[0]
      if verbose:
        print(f"Selecting most recent job: {target_job['job_id']} (use job-id to view others)", file=sys.stderr)

  log_file = target_job.get("log_file")
  if not log_file:
    raise ValueError(f"Log file not specified in metadata for job {target_job.get('job_id')}")

  status = target_job.get("status")
  pid = target_job.get("pid")

  if follow:
    n = tail_lines if tail_lines else 50
    if status and status != "RUNNING":
      if verbose:
        print(f"Job '{target_job.get('job_id')}' has finished with status: {status} (exit code: {target_job.get('exit_code', 'unknown')}).", file=sys.stderr)
      remote_cmd = f"tail -n {n} '{log_file}'"
    else:
      if pid:
        tail_part = f"tail -n {n} -s 0.2 --pid={pid} -f '{log_file}' 2>/dev/null || tail -n {n} -f '{log_file}'"
      else:
        tail_part = f"tail -n {n} -f '{log_file}'"
      remote_cmd = f"bash -c '{tail_part} & TPID=$!; trap \"kill -9 $TPID 2>/dev/null\" EXIT INT TERM HUP; wait $TPID'"
  elif tail_lines:
    remote_cmd = f"tail -n {tail_lines} '{log_file}'"
  else:
    remote_cmd = f"cat '{log_file}'"

  cmd = build_ssh_cmd(
      host,
      port,
      remote_cmd,
      private_key_path=private_key_path,
      tty=False,
      ssh_config_path=ssh_config_path
  )
  try:
    subprocess.run(cmd)
  except KeyboardInterrupt:
    if verbose:
      print("\nDetached from log stream.", file=sys.stderr)


def kill_remote_job(host, port, target_id, signal_name="SIGTERM", timeout=15.0, private_key_path=None, ssh_config_path=None):
  ensure_remote_runner(host, port, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  kill_cmd = f"python3 {REMOTE_RUNNER_PATH} kill --target {shlex.quote(str(target_id))} --signal {shlex.quote(signal_name)} --timeout {float(timeout)}"
  cmd = build_ssh_cmd(host, port, kill_cmd, private_key_path=private_key_path, ssh_config_path=ssh_config_path)
  res = subprocess.run(cmd, capture_output=True, text=True)
  output = res.stdout.strip()

  if "KILLED_SIGKILL" in output:
    print(f"Graceful termination timed out after {timeout:.1f}s; sent SIGKILL to process {target_id} and its process group.")
  elif "KILLED" in output:
    print(f"Signal {signal_name} sent to process {target_id} and its process group.")
  elif "NOT_RUNNING" in output:
    print(f"Process {target_id} was not running.")
  else:
    print(f"Result: {output}")
