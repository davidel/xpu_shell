#!/usr/bin/env python3
"""Remote job supervisor and process manager for XPU Shell.

This script runs on the remote instance (RunPod, Vast.ai, etc.) to supervise jobs,
record environment snapshots, monitor process exit status, detect kernel OOM
events, and manage job lifecycles.
"""


import argparse
import base64
import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


def get_base_dir():
  if os.path.isdir("/workspace"):
    return Path("/workspace")
  return Path("/tmp")


def get_jobs_dirs():
  return [Path("/workspace/.runpod_jobs"), Path("/tmp/.runpod_jobs")]


def allocate_job_id(prefix="job-", base_dir=None):
  if base_dir is None:
    target_base = get_base_dir()
  else:
    target_base = Path(base_dir)

  jobs_dir = target_base / ".runpod_jobs" if target_base.name != ".runpod_jobs" else target_base
  jobs_dir.mkdir(parents=True, exist_ok=True)
  lock_file_path = jobs_dir / ".counter.lock"
  counter_file_path = jobs_dir / ".counter"

  with open(lock_file_path, "w") as lock_file:
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
      current_id = 0
      if counter_file_path.exists():
        try:
          current_id = int(counter_file_path.read_text().strip())
        except (ValueError, OSError):
          current_id = 0

      if current_id == 0:
        max_existing = 0
        search_dirs = [jobs_dir] if base_dir else get_jobs_dirs()
        for b_dir in search_dirs:
          if b_dir.exists():
            try:
              for entry in b_dir.iterdir():
                if entry.is_dir():
                  name = entry.name
                  if name.startswith(prefix):
                    num_part = name[len(prefix):]
                    if num_part.isdigit():
                      max_existing = max(max_existing, int(num_part))
                  elif name.isdigit():
                    max_existing = max(max_existing, int(name))
            except OSError:
              pass
        current_id = max(current_id, max_existing)

      next_id = current_id + 1
      counter_file_path.write_text(str(next_id))
      return f"{prefix}{next_id}"
    finally:
      try:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
      except OSError:
        pass


def cmd_next_id(args):
  prefix = getattr(args, "prefix", "job-") or "job-"
  base_dir = getattr(args, "base_dir", None)
  jid = allocate_job_id(prefix=prefix, base_dir=base_dir)
  print(jid)


def format_duration(seconds):
  seconds = max(0, int(seconds))
  mins, secs = divmod(seconds, 60)
  hours, mins = divmod(mins, 60)
  if hours > 0:
    return f"{hours}h {mins}m {secs}s"
  return f"{mins}m {secs}s"


def is_pid_alive(pid):
  try:
    p_int = int(pid)
    if p_int <= 1:
      return False
    os.kill(p_int, 0)
  except (OSError, ValueError, TypeError):
    return False
  try:
    with open(f"/proc/{p_int}/status", "r") as f:
      for line in f:
        if line.startswith("State:"):
          return "Z" not in line and "X" not in line
  except (IOError, OSError):
    pass
  return True


def is_pgid_alive(pgid):
  if pgid <= 1:
    return False
  if not os.path.exists("/proc"):
    try:
      os.killpg(pgid, 0)
      return True
    except OSError:
      return False
  try:
    for entry in os.scandir("/proc"):
      if entry.name.isdigit():
        try:
          with open(os.path.join(entry.path, "stat"), "r") as f:
            content = f.read()
          idx = content.rfind(")")
          if idx != -1:
            rest = content[idx + 2:].split()
            state = rest[0]
            pgrp = int(rest[2])
            if pgrp == pgid and state not in ("Z", "X"):
              return True
        except (IOError, OSError, IndexError, ValueError):
          continue
  except OSError:
    pass
  return False


def cmd_run(args):
  job_id = getattr(args, "job_id", None)
  if not job_id:
    job_id = allocate_job_id()

  base_dir = get_base_dir()
  if getattr(args, "job_dir", None):
    job_dir = Path(args.job_dir)
  else:
    job_dir = base_dir / ".runpod_jobs" / job_id
  job_dir.mkdir(parents=True, exist_ok=True)

  if getattr(args, "log_file", None):
    log_file = Path(args.log_file)
  else:
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}.log"
  log_file.parent.mkdir(parents=True, exist_ok=True)

  started_at = int(time.time())
  now_utc = datetime.datetime.now(datetime.timezone.utc)
  started_at_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
  started_at_human = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

  pid = os.getpid()

  work_dir = get_base_dir()
  if args.work_dir and os.path.isdir(args.work_dir):
    work_dir = Path(args.work_dir)

  python_executable = shutil.which("python3") or shutil.which("python") or "python3"

  if getattr(args, "cmd", None):
    if getattr(args, "shell", False):
      shell_binary = os.environ.get("SHELL") or shutil.which("bash") or "/bin/sh"
      full_cmd_str = args.cmd
      if args.args:
        full_cmd_str = f"{full_cmd_str} {args.args}"
      cmd = [shell_binary, "-c", full_cmd_str]
      script_name = Path(shell_binary).name
      script_path_str = f"{shell_binary} -c {shlex.quote(full_cmd_str)}"
    else:
      cmd = shlex.split(args.cmd)
      if args.args:
        cmd.extend(shlex.split(args.args))
      script_name = Path(cmd[0]).name if cmd else "command"
      script_path_str = " ".join(shlex.quote(c) for c in cmd)
  elif getattr(args, "script", None):
    script_path = Path(args.script)
    script_name = getattr(args, "name", None) or re.sub(r"^upload_\d+_[a-f0-9]+_", "", script_path.name)
    script_path_str = args.script
    if not script_path.exists():
      print(f"ERROR: Script not found at {script_path}", file=sys.stderr)
      sys.stderr.flush()
      (job_dir / "exit_code").write_text("127")
      meta = {
          "job_id": job_id,
          "pid": pid,
          "script": script_name,
          "args": args.args or "",
          "started_at": started_at,
          "started_at_iso": started_at_iso,
          "log_file": str(log_file),
          "script_path": script_path_str,
          "status": "FAILED(127)",
          "ended_at": int(time.time())
      }
      meta_file = job_dir / "meta.json"
      meta_file.write_text(json.dumps(meta, indent=2))
      sys.exit(127)

    try:
      os.chmod(script_path, 0o755)
    except OSError:
      pass

    if script_path.suffix == ".py":
      cmd = [python_executable, str(script_path)]
    elif script_path.suffix == ".sh":
      cmd = ["/bin/bash", str(script_path)]
    else:
      cmd = [str(script_path)]

    if args.args:
      cmd.extend(shlex.split(args.args))
  else:
    print("ERROR: Neither --script nor --cmd specified", file=sys.stderr)
    sys.stderr.flush()
    (job_dir / "exit_code").write_text("1")
    sys.exit(1)

  meta = {
      "job_id": job_id,
      "pid": pid,
      "script": script_name,
      "args": args.args or "",
      "started_at": started_at,
      "started_at_iso": started_at_iso,
      "log_file": str(log_file),
      "script_path": script_path_str,
      "status": "RUNNING"
  }
  meta_file = job_dir / "meta.json"
  meta_file.write_text(json.dumps(meta, indent=2))
  (job_dir / "pid").write_text(str(pid))

  # Prepare environment
  env = os.environ.copy()
  env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
  env["PYTHONUNBUFFERED"] = "1"
  job_env_b64 = env.pop("RUNPOD_JOB_ENV", None)
  if job_env_b64:
    try:
      job_env_data = json.loads(base64.b64decode(job_env_b64).decode("utf-8"))
      if isinstance(job_env_data, dict):
        env.update(job_env_data)
    except Exception as e:
      print(f"WARNING: Failed to decode RUNPOD_JOB_ENV payload: {e}", file=sys.stderr)

  verbose = getattr(args, "verbose", False)
  if verbose:
    # Log job start header
    print("=" * 80, file=sys.stderr)
    print(f"=== RUNPOD JOB STARTED: {job_id}", file=sys.stderr)
    print(f"=== Start Time:  {started_at_human}", file=sys.stderr)
    cmd_display = script_path_str if getattr(args, "cmd", None) else f"{args.script} {args.args or ''}".strip()
    print(f"=== Command:     {cmd_display}", file=sys.stderr)
    print(f"=== Working Dir: {work_dir}", file=sys.stderr)
    print(f"=== Python:      {python_executable}", file=sys.stderr)

    # Check GPU visibility
    try:
      gpu_res = subprocess.run(
          ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
          capture_output=True,
          text=True,
          timeout=3
      )
      if gpu_res.returncode == 0 and gpu_res.stdout.strip():
        for line in gpu_res.stdout.strip().splitlines():
          print(f"=== GPU:         {line.strip()}", file=sys.stderr)
    except Exception:
      pass

    print("=" * 80, file=sys.stderr)
    sys.stderr.flush()

  child = None
  try:
    child = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        env=env,
        preexec_fn=os.setpgrp
    )
  except Exception as e:
    print(f"ERROR: Failed to launch child process: {e}", file=sys.stderr)
    sys.stderr.flush()
    exit_val = 127 if isinstance(e, FileNotFoundError) else 1
    (job_dir / "exit_code").write_text(str(exit_val))
    meta["status"] = f"FAILED({exit_val})"
    meta["ended_at"] = int(time.time())
    meta_file.write_text(json.dumps(meta, indent=2))
    sys.exit(exit_val)

  (job_dir / "child_pid").write_text(str(child.pid))
  meta["child_pid"] = child.pid
  meta_file.write_text(json.dumps(meta, indent=2))

  # Forward termination signals to child process group
  def handle_signal(signum, frame):
    if child and child.poll() is None:
      try:
        os.killpg(child.pid, signum)
      except OSError:
        pass

  signal.signal(signal.SIGTERM, handle_signal)
  signal.signal(signal.SIGINT, handle_signal)
  if hasattr(signal, "SIGHUP"):
    signal.signal(signal.SIGHUP, handle_signal)

  exit_code = child.wait()

  ended_at = int(time.time())
  end_utc = datetime.datetime.now(datetime.timezone.utc)
  ended_at_iso = end_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
  ended_at_human = end_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
  dur_secs = max(0, ended_at - started_at)
  dur_str = format_duration(dur_secs)

  (job_dir / "exit_code").write_text(str(exit_code))

  if verbose:
    # Log job end footer
    print("\n" + "=" * 80, file=sys.stderr)
    print(f"=== RUNPOD JOB COMPLETED: {job_id}", file=sys.stderr)
    print(f"=== End Time:    {ended_at_human}", file=sys.stderr)
    print(f"=== Duration:    {dur_str}", file=sys.stderr)
    print(f"=== Exit Code:   {exit_code}", file=sys.stderr)

  if exit_code in (137, -9):
    print("=== NOTICE: Exit code 137 indicates process was killed via SIGKILL (often Linux OOM Killer).", file=sys.stderr)
    try:
      dmesg_res = subprocess.run(
          ["dmesg", "-T"],
          capture_output=True,
          text=True,
          timeout=3
      )
      oom_lines = [
          l for l in dmesg_res.stdout.splitlines()
          if any(k in l.lower() for k in ("oom-killer", "out of memory", "killed process"))
      ]
      if oom_lines:
        print("=== Kernel OOM Killer events detected in dmesg:", file=sys.stderr)
        for l in oom_lines[-5:]:
          print(f"    {l}", file=sys.stderr)
    except Exception:
      pass

  if verbose:
    print("=" * 80, file=sys.stderr)
    sys.stderr.flush()

  status = "COMPLETED" if exit_code == 0 else f"FAILED({exit_code})"
  if (job_dir / "killed").exists():
    status = "KILLED"

  meta["status"] = status
  meta["exit_code"] = exit_code
  meta["ended_at"] = ended_at
  meta["ended_at_iso"] = ended_at_iso
  meta["duration"] = dur_str
  meta["duration_seconds"] = dur_secs
  meta_file.write_text(json.dumps(meta, indent=2))

  sys.exit(exit_code)


def cmd_spawn(args):
  base_dir = Path(args.base_dir) if getattr(args, "base_dir", None) else get_base_dir()
  job_id = getattr(args, "job_id", None)
  if not job_id:
    job_id = allocate_job_id(base_dir=base_dir)

  if getattr(args, "job_dir", None):
    job_dir = Path(args.job_dir)
  else:
    job_dir = base_dir / ".runpod_jobs" / job_id
  job_dir.mkdir(parents=True, exist_ok=True)

  name = getattr(args, "name", None)
  if not name:
    if getattr(args, "script", None):
      script_base = Path(args.script).name
      name = re.sub(r"^upload_\d+_[a-f0-9]+_", "", script_base)
    elif getattr(args, "cmd", None):
      if getattr(args, "shell", False):
        name = "shell"
      else:
        tokens = shlex.split(args.cmd)
        name = Path(tokens[0]).name if tokens else "cmd"
    else:
      name = "job"
  name = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', name) or "job"

  if getattr(args, "log_file", None):
    log_file = Path(args.log_file)
  else:
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"{job_id}_{name}.log"
  log_file.parent.mkdir(parents=True, exist_ok=True)

  runner_script = str(Path(__file__).resolve())
  run_cmd = [
      sys.executable,
      runner_script,
      "run",
      "--job-id", str(job_id),
      "--job-dir", str(job_dir),
      "--log-file", str(log_file)
  ]
  if getattr(args, "script", None):
    run_cmd.extend(["--script", str(args.script)])
  if name:
    run_cmd.extend(["--name", str(name)])
  if getattr(args, "cmd", None):
    run_cmd.extend(["--cmd", str(args.cmd)])
  if getattr(args, "args", None):
    run_cmd.extend(["--args", str(args.args)])
  if getattr(args, "work_dir", None):
    run_cmd.extend(["--work-dir", str(args.work_dir)])
  if getattr(args, "shell", False):
    run_cmd.append("--shell")
  if getattr(args, "verbose", False):
    run_cmd.append("--verbose")

  env = os.environ.copy()
  env["PIP_BREAK_SYSTEM_PACKAGES"] = "1"
  env["PYTHONUNBUFFERED"] = "1"

  log_fd = open(log_file, "a")
  try:
    proc = subprocess.Popen(
        run_cmd,
        stdin=subprocess.DEVNULL,
        stdout=log_fd,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
        env=env
    )
  finally:
    log_fd.close()

  (job_dir / "pid").write_text(str(proc.pid))

  print(f"PID:{proc.pid}")
  print(f"LOG_FILE:{log_file}")
  print(f"JOB_ID:{job_id}")
  sys.stdout.flush()
  sys.exit(0)


def cmd_list(args):
  jobs = []
  now = int(time.time())
  base_dirs = [Path(args.base_dir)] if getattr(args, "base_dir", None) else get_jobs_dirs()
  seen_job_ids = set()

  for base_dir in base_dirs:
    if not base_dir.exists():
      continue
    for entry in base_dir.iterdir():
      if not entry.is_dir():
        continue
      meta_file = entry / "meta.json"
      if not meta_file.exists():
        continue
      try:
        data = json.loads(meta_file.read_text())
      except Exception:
        continue

      job_id = data.get("job_id", entry.name)
      if job_id in seen_job_ids:
        continue
      seen_job_ids.add(job_id)

      pid = data.get("pid")
      child_pid = data.get("child_pid")
      is_running = False
      if pid and is_pid_alive(pid):
        is_running = True
      elif child_pid and is_pid_alive(child_pid):
        is_running = True

      exit_code_file = entry / "exit_code"
      killed_file = entry / "killed"

      if is_running:
        status = "RUNNING"
      elif killed_file.exists():
        status = "KILLED"
      elif exit_code_file.exists():
        try:
          ec = exit_code_file.read_text().strip()
          status = "COMPLETED" if ec == "0" else f"FAILED({ec})"
        except Exception:
          status = "EXITED"
      else:
        status = data.get("status", "EXITED")

      data["status"] = status
      started = data.get("started_at", now)

      if is_running:
        dur_secs = now - started
      elif "ended_at" in data:
        dur_secs = data["ended_at"] - started
      elif exit_code_file.exists():
        try:
          dur_secs = int(exit_code_file.stat().st_mtime) - started
        except OSError:
          dur_secs = data.get("duration_seconds", now - started)
      else:
        dur_secs = data.get("duration_seconds", now - started)

      data["duration"] = format_duration(dur_secs)
      jobs.append(data)

  jobs.sort(key=lambda x: x.get("started_at", 0), reverse=True)
  print(json.dumps(jobs))


def cmd_kill(args):
  target_id = str(args.target)
  sig_name = args.signal or "SIGTERM"
  try:
    sig_num = getattr(signal, sig_name)
  except AttributeError:
    sig_num = signal.SIGTERM

  target_job_dir = None
  target_pid = None
  target_child_pid = None
  base_dirs = [Path(args.base_dir)] if getattr(args, "base_dir", None) else get_jobs_dirs()

  for base_dir in base_dirs:
    if not base_dir.exists():
      continue
    for entry in base_dir.iterdir():
      if entry.name in (target_id, f"job-{target_id}"):
        target_job_dir = entry
        pid_file = entry / "pid"
        if pid_file.exists():
          target_pid = pid_file.read_text().strip()
        child_pid_file = entry / "child_pid"
        if child_pid_file.exists():
          target_child_pid = child_pid_file.read_text().strip()
        break
      meta_file = entry / "meta.json"
      if meta_file.exists():
        try:
          data = json.loads(meta_file.read_text())
          if str(data.get("pid")) == target_id or data.get("job_id") in (target_id, f"job-{target_id}") or str(data.get("child_pid")) == target_id:
            target_job_dir = entry
            target_pid = str(data.get("pid"))
            if "child_pid" in data:
              target_child_pid = str(data.get("child_pid"))
            break
        except Exception:
          pass
    if target_job_dir:
      break

  timeout = float(getattr(args, "timeout", 15.0) or 15.0)

  if not target_pid:
    target_pid = target_id

  pids_to_kill = []
  for p_str in (target_child_pid, target_pid):
    if not p_str:
      continue
    try:
      p_int = int(p_str)
      if p_int > 1 and p_int not in pids_to_kill:
        pids_to_kill.append(p_int)
    except ValueError:
      continue

  my_pgid = os.getpgrp()

  pgids_to_kill = set()
  for p in pids_to_kill:
    try:
      pgid = os.getpgid(p)
      if pgid > 1 and pgid != my_pgid:
        pgids_to_kill.add(pgid)
    except OSError:
      pass

  if target_child_pid:
    try:
      c_int = int(target_child_pid)
      if c_int > 1 and c_int != my_pgid:
        pgids_to_kill.add(c_int)
    except ValueError:
      pass

  def is_any_alive():
    for p in pids_to_kill:
      if is_pid_alive(p):
        return True
    for pgid in pgids_to_kill:
      if is_pgid_alive(pgid):
        return True
    return False

  def send_to_targets(signum):
    sent = False
    for pgid in pgids_to_kill:
      try:
        os.killpg(pgid, signum)
        sent = True
      except OSError:
        pass
    for p in pids_to_kill:
      if p != os.getpid():
        try:
          os.kill(p, signum)
          sent = True
        except OSError:
          pass
    return sent

  if not is_any_alive():
    print("NOT_RUNNING")
    return

  send_to_targets(sig_num)
  killed_via_sigkill = False

  # If signal was SIGTERM, wait up to timeout seconds, then escalate to SIGKILL
  if sig_name in ("SIGTERM", "15") and timeout > 0:
    start_wait = time.time()
    while time.time() - start_wait < timeout:
      if not is_any_alive():
        break
      time.sleep(0.2)
    else:
      if is_any_alive():
        send_to_targets(signal.SIGKILL)
        killed_via_sigkill = True
        time.sleep(0.1)

  if target_job_dir and target_job_dir.exists():
    try:
      (target_job_dir / "killed").touch()
      meta_f = target_job_dir / "meta.json"
      if meta_f.exists():
        data = json.loads(meta_f.read_text())
        lf = data.get("log_file")
        if lf:
          log_escaped = shlex.quote(str(lf))
          subprocess.run(
              f"pkill -f 'tail .* {log_escaped}'",
              shell=True,
              capture_output=True
          )
    except Exception:
      pass

  if killed_via_sigkill:
    print("KILLED_SIGKILL")
  else:
    print("KILLED")


def main():
  parser = argparse.ArgumentParser(description="XPU Remote Job Runner")
  subparsers = parser.add_subparsers(dest="command", required=True)

  run_p = subparsers.add_parser("run")
  run_p.add_argument("--job-id", default=None)
  run_p.add_argument("--script", default=None)
  run_p.add_argument("--cmd", default=None)
  run_p.add_argument("--args", default="")
  run_p.add_argument("--job-dir", default=None)
  run_p.add_argument("--log-file", default=None)
  run_p.add_argument("--work-dir", default="")
  run_p.add_argument("--shell", action="store_true", default=False)
  run_p.add_argument("--name", default=None)
  run_p.add_argument("-v", "--verbose", action="store_true", default=False)

  next_id_p = subparsers.add_parser("next-id", help="Allocate the next sequential job ID")
  next_id_p.add_argument("--prefix", default="job-")
  next_id_p.add_argument("--base-dir", default=None)

  list_p = subparsers.add_parser("list")
  list_p.add_argument("--base-dir", default=None)

  kill_p = subparsers.add_parser("kill")
  kill_p.add_argument("--target", required=True)
  kill_p.add_argument("--signal", default="SIGTERM")
  kill_p.add_argument("--timeout", type=float, default=15.0)
  kill_p.add_argument("--base-dir", default=None)

  spawn_p = subparsers.add_parser("spawn", help="Spawn a detached job daemon in a new session")
  spawn_p.add_argument("--job-id", default=None)
  spawn_p.add_argument("--script", default=None)
  spawn_p.add_argument("--cmd", default=None)
  spawn_p.add_argument("--args", default="")
  spawn_p.add_argument("--name", default=None)
  spawn_p.add_argument("--job-dir", default=None)
  spawn_p.add_argument("--log-file", default=None)
  spawn_p.add_argument("--work-dir", default="")
  spawn_p.add_argument("--base-dir", default=None)
  spawn_p.add_argument("--shell", action="store_true", default=False)
  spawn_p.add_argument("-v", "--verbose", action="store_true", default=False)

  args = parser.parse_args()
  if args.command == "run":
    cmd_run(args)
  elif args.command == "spawn":
    cmd_spawn(args)
  elif args.command == "list":
    cmd_list(args)
  elif args.command == "kill":
    cmd_kill(args)
  elif args.command == "next-id":
    cmd_next_id(args)


if __name__ == "__main__":
  main()
