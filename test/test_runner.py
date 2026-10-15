import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import xpu_shell.runner as runner


RUNNER_PATH = Path(__file__).parent.parent / "src" / "xpu_shell" / "runner.py"



class TestRunnerScript(unittest.TestCase):

  def test_runner_run_and_list_success(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      script = tdp / "job.py"
      script.write_text("import time\nprint('starting task')\ntime.sleep(0.5)\nprint('finished task')\n")

      job_dir = tdp / ".runpod_jobs" / "test-job-1"
      log_file = tdp / "logs" / "test-job-1.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        res = subprocess.run([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "test-job-1",
            "--script", str(script),
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp),
            "-v"
        ], stdout=lf, stderr=subprocess.STDOUT)

      self.assertEqual(res.returncode, 0)
      log_content = log_file.read_text()
      self.assertIn("=== RUNPOD JOB STARTED: test-job-1", log_content)
      self.assertIn("starting task", log_content)
      self.assertIn("finished task", log_content)
      self.assertIn("=== RUNPOD JOB COMPLETED: test-job-1", log_content)
      self.assertIn("=== Exit Code:   0", log_content)

      # Verify exit_code and pid files
      self.assertEqual((job_dir / "exit_code").read_text().strip(), "0")
      self.assertTrue((job_dir / "pid").exists())
      self.assertTrue((job_dir / "child_pid").exists())

      # Verify list
      list_res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "list",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)

      self.assertEqual(list_res.returncode, 0)
      jobs = json.loads(list_res.stdout)
      self.assertEqual(len(jobs), 1)
      self.assertEqual(jobs[0]["job_id"], "test-job-1")
      self.assertEqual(jobs[0]["status"], "COMPLETED")
      self.assertIn("duration", jobs[0])

  def test_runner_run_missing_script(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      missing_script = tdp / "does_not_exist.py"
      job_dir = tdp / ".runpod_jobs" / "missing-job"
      log_file = tdp / "logs" / "missing.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        res = subprocess.run([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "missing-job",
            "--script", str(missing_script),
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp)
        ], stdout=lf, stderr=subprocess.STDOUT)

      self.assertEqual(res.returncode, 127)
      self.assertEqual((job_dir / "exit_code").read_text().strip(), "127")

      list_res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "list",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)

      jobs = json.loads(list_res.stdout)
      self.assertEqual(len(jobs), 1)
      self.assertEqual(jobs[0]["status"], "FAILED(127)")

  def test_runner_kill(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      script = tdp / "sleep_job.py"
      script.write_text("import time\nprint('sleeping')\ntime.sleep(60)\n")

      job_dir = tdp / ".runpod_jobs" / "kill-job"
      log_file = tdp / "logs" / "kill-job.log"
      job_dir.mkdir(parents=True, exist_ok=True)
      log_file.parent.mkdir(parents=True, exist_ok=True)

      lf = open(log_file, "w")
      proc = subprocess.Popen([
          sys.executable,
          str(RUNNER_PATH),
          "run",
          "--job-id", "kill-job",
          "--script", str(script),
          "--job-dir", str(job_dir),
          "--log-file", str(log_file),
          "--work-dir", str(tdp)
      ], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

      time.sleep(0.5)

      # Check running
      list_res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "list",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)
      jobs = json.loads(list_res.stdout)
      self.assertEqual(jobs[0]["status"], "RUNNING")

      # Kill
      kill_res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "kill",
          "--target", "kill-job",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)

      self.assertEqual(kill_res.stdout.strip(), "KILLED")
      proc.wait(timeout=5)
      lf.close()

      # Check killed status
      list_res2 = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "list",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)
      jobs2 = json.loads(list_res2.stdout)
      self.assertEqual(jobs2[0]["status"], "KILLED")

  def test_runner_kill_timeout_escalation(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      script = tdp / "stubborn.py"
      script.write_text(
          "#!/usr/bin/env python3\n"
          "import signal, time\n"
          "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
          "time.sleep(30)\n"
      )

      job_dir = tdp / ".runpod_jobs" / "kill-stubborn"
      log_file = tdp / "logs" / "kill-stubborn.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        proc = subprocess.Popen([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "kill-stubborn",
            "--script", str(script),
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp)
        ], stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

      time.sleep(0.5)

      # Check running
      list_res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "list",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)
      jobs = json.loads(list_res.stdout)
      self.assertEqual(jobs[0]["status"], "RUNNING")

      # Kill with 0.5s timeout to trigger SIGKILL escalation
      kill_res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "kill",
          "--target", "kill-stubborn",
          "--timeout", "0.5",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)

      self.assertEqual(kill_res.stdout.strip(), "KILLED_SIGKILL")
      proc.wait(timeout=5)

      # Check killed status
      list_res2 = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "list",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)
      jobs2 = json.loads(list_res2.stdout)
      self.assertEqual(jobs2[0]["status"], "KILLED")

  def test_runner_run_script_without_shebang(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      script = tdp / "noshebang.py"
      # Pure python code without #!/usr/bin/env python3
      script.write_text("print('python executed without shebang')\n")

      job_dir = tdp / ".runpod_jobs" / "test-shebang-free"
      log_file = tdp / "logs" / "test-shebang-free.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        res = subprocess.run([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "test-shebang-free",
            "--script", str(script),
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp)
        ], stdout=lf, stderr=subprocess.STDOUT)

      self.assertEqual(res.returncode, 0)
      log_content = log_file.read_text()
      self.assertEqual(log_content.strip(), "python executed without shebang")

  def test_runner_oom_notice(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      script = tdp / "oom_job.py"
      script.write_text("import sys\nsys.exit(137)\n")

      job_dir = tdp / ".runpod_jobs" / "test-oom"
      log_file = tdp / "logs" / "test-oom.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        res = subprocess.run([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "test-oom",
            "--script", str(script),
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp)
        ], stdout=lf, stderr=subprocess.STDOUT)

      self.assertEqual(res.returncode, 137)
      log_content = log_file.read_text()
      self.assertIn("Exit code 137 indicates process was killed via SIGKILL", log_content)

  def test_runner_run_direct_command(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      job_dir = tdp / ".runpod_jobs" / "test-cmd-1"
      log_file = tdp / "logs" / "test-cmd-1.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        res = subprocess.run([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "test-cmd-1",
            "--cmd", f"{sys.executable} -c \"print('direct command output')\"",
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp),
            "-v"
        ], stdout=lf, stderr=subprocess.STDOUT)

      self.assertEqual(res.returncode, 0)
      log_content = log_file.read_text()
      self.assertIn("=== RUNPOD JOB STARTED: test-cmd-1", log_content)
      self.assertIn("direct command output", log_content)
      self.assertIn("=== RUNPOD JOB COMPLETED: test-cmd-1", log_content)
      self.assertIn("=== Exit Code:   0", log_content)

      self.assertEqual((job_dir / "exit_code").read_text().strip(), "0")
      meta = json.loads((job_dir / "meta.json").read_text())
      self.assertEqual(meta["job_id"], "test-cmd-1")
      self.assertEqual(meta["status"], "COMPLETED")

  def test_runner_run_shell_mode(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      (tdp / "alpha_123.txt").write_text("alpha content\n")
      (tdp / "alpha_456.txt").write_text("another alpha\n")

      job_dir = tdp / ".runpod_jobs" / "test-shell-1"
      log_file = tdp / "logs" / "test-shell-1.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      with open(log_file, "w") as lf:
        res = subprocess.run([
            sys.executable,
            str(RUNNER_PATH),
            "run",
            "--job-id", "test-shell-1",
            "--shell",
            "--cmd", "ls alpha_*.txt | wc -l",
            "--job-dir", str(job_dir),
            "--log-file", str(log_file),
            "--work-dir", str(tdp),
            "-v"
        ], stdout=lf, stderr=subprocess.STDOUT)

      self.assertEqual(res.returncode, 0)
      log_content = log_file.read_text()
      self.assertIn("=== RUNPOD JOB STARTED: test-shell-1", log_content)
      # Two matching files: wc -l should output 2
      self.assertIn("2", log_content)
      self.assertIn("=== RUNPOD JOB COMPLETED: test-shell-1", log_content)
      self.assertEqual((job_dir / "exit_code").read_text().strip(), "0")

  def test_allocate_job_id_sequential(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      id1 = runner.allocate_job_id(base_dir=tdp)
      id2 = runner.allocate_job_id(base_dir=tdp)
      id3 = runner.allocate_job_id(base_dir=tdp)
      self.assertEqual(id1, "job-1")
      self.assertEqual(id2, "job-2")
      self.assertEqual(id3, "job-3")

  def test_allocate_job_id_directory_fallback(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      jobs_dir = tdp / ".runpod_jobs"
      jobs_dir.mkdir(parents=True, exist_ok=True)
      (jobs_dir / "job-1").mkdir()
      (jobs_dir / "job-5").mkdir()
      (jobs_dir / "job-12").mkdir()

      next_id = runner.allocate_job_id(base_dir=tdp)
      self.assertEqual(next_id, "job-13")

  def test_allocate_job_id_concurrent(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      src_dir = Path(__file__).parent.parent / "src"
      worker_code = (
          f"import sys\n"
          f"sys.path.insert(0, '{src_dir}')\n"
          f"import xpu_shell.runner as r\n"
          f"print(r.allocate_job_id(base_dir='{tdp}'))\n"

      )

      procs = [
          subprocess.Popen([sys.executable, "-c", worker_code], stdout=subprocess.PIPE, text=True)
          for _ in range(10)
      ]
      results = [p.communicate()[0].strip() for p in procs]
      for p in procs:
        self.assertEqual(p.returncode, 0)

      self.assertEqual(len(results), 10)
      self.assertEqual(len(set(results)), 10)
      expected = {f"job-{i}" for i in range(1, 11)}
      self.assertEqual(set(results), expected)

  def test_runner_next_id_cli(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      res1 = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "next-id",
          "--base-dir", str(tdp)
      ], capture_output=True, text=True)
      self.assertEqual(res1.returncode, 0)
      self.assertEqual(res1.stdout.strip(), "job-1")

      res2 = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "next-id",
          "--base-dir", str(tdp)
      ], capture_output=True, text=True)
      self.assertEqual(res2.returncode, 0)
      self.assertEqual(res2.stdout.strip(), "job-2")

  def test_runner_run_auto_allocated_job_id(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      jobs_dir = tdp / ".runpod_jobs"
      jobs_dir.mkdir(parents=True, exist_ok=True)
      log_file = tdp / "logs" / "auto.log"
      log_file.parent.mkdir(parents=True, exist_ok=True)

      res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "run",
          "--cmd", "echo auto-job",
          "--job-dir", str(jobs_dir / "job-1"),
          "--log-file", str(log_file),
          "--work-dir", str(tdp)
      ], capture_output=True, text=True)
      self.assertEqual(res.returncode, 0)
      meta = json.loads((jobs_dir / "job-1" / "meta.json").read_text())
      self.assertTrue(meta["job_id"].startswith("job-"))

  def test_runner_kill_with_shorthand_id(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      job_dir = tdp / ".runpod_jobs" / "job-7"
      job_dir.mkdir(parents=True, exist_ok=True)
      (job_dir / "meta.json").write_text(json.dumps({"job_id": "job-7", "pid": 999999}))
      (job_dir / "pid").write_text("999999")

      res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "kill",
          "--target", "7",
          "--base-dir", str(tdp / ".runpod_jobs")
      ], capture_output=True, text=True)
      self.assertEqual(res.returncode, 0)
      self.assertEqual(res.stdout.strip(), "NOT_RUNNING")

  def test_runner_spawn_cmd(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "spawn",
          "--cmd", "echo spawn-success",
          "--work-dir", str(tdp)
      ], capture_output=True, text=True)

      self.assertEqual(res.returncode, 0)
      output = res.stdout
      self.assertIn("PID:", output)
      self.assertIn("LOG_FILE:", output)
      self.assertIn("JOB_ID:job-", output)

      log_file_line = [l for l in output.splitlines() if l.startswith("LOG_FILE:")][0]
      log_file = Path(log_file_line.split("LOG_FILE:", 1)[1].strip())

      for _ in range(50):
        if log_file.exists() and "spawn-success" in log_file.read_text():
          break
        time.sleep(0.1)
      self.assertTrue(log_file.exists())
      self.assertIn("spawn-success", log_file.read_text())
      # Clean up log file
      try:
        log_file.unlink()
      except OSError:
        pass

  def test_runner_spawn_script_strips_upload_prefix(self):
    with tempfile.TemporaryDirectory() as td:
      tdp = Path(td)
      script_file = tdp / "upload_1789019711_939366_finetune_uvito.sh"
      script_file.write_text("#!/bin/bash\necho finished\n")
      script_file.chmod(0o755)

      res = subprocess.run([
          sys.executable,
          str(RUNNER_PATH),
          "spawn",
          "--script", str(script_file),
          "--work-dir", str(tdp),
          "--base-dir", str(tdp)
      ], capture_output=True, text=True)

      self.assertEqual(res.returncode, 0)
      output = res.stdout
      log_file_line = [l for l in output.splitlines() if l.startswith("LOG_FILE:")][0]
      log_file = Path(log_file_line.split("LOG_FILE:", 1)[1].strip())
      self.assertEqual(log_file.name, "job-1_finetune_uvito.sh.log")

      for _ in range(50):
        if log_file.exists() and "finished" in log_file.read_text():
          break
        time.sleep(0.1)
      self.assertTrue(log_file.exists())
      try:
        log_file.unlink()
      except OSError:
        pass


if __name__ == "__main__":
  unittest.main()
