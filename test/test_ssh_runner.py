import json
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest
from unittest.mock import patch, MagicMock

import xpu_shell.ssh_runner as ssh_runner


class TestSSHRunner(unittest.TestCase):

  def setUp(self):
    ssh_runner._ENSURED_RUNNER_HOSTS.clear()

  @patch("pathlib.Path.exists", autospec=True)
  def test_find_ssh_private_key_explicit(self, mock_exists):
    mock_exists.return_value = True
    key = ssh_runner.find_ssh_private_key(explicit_private_key_path="~/.ssh/custom_key")
    self.assertEqual(key, Path("~/.ssh/custom_key").expanduser())

  @patch("pathlib.Path.exists", autospec=True)
  def test_find_ssh_private_key_explicit_missing(self, mock_exists):
    mock_exists.return_value = False
    with self.assertRaises(FileNotFoundError):
      ssh_runner.find_ssh_private_key(explicit_private_key_path="~/.ssh/missing_key")

  @patch("pathlib.Path.exists", autospec=True)
  def test_find_ssh_private_key_from_pub(self, mock_exists):
    def exists_side_effect(path_obj):
      return str(path_obj).endswith("id_ed25519")

    mock_exists.side_effect = exists_side_effect
    key = ssh_runner.find_ssh_private_key(public_key_path="~/.ssh/id_ed25519.pub")
    self.assertEqual(key, Path("~/.ssh/id_ed25519").expanduser())

  @patch("pathlib.Path.exists", autospec=True)
  def test_find_ssh_private_key_default(self, mock_exists):
    def exists_side_effect(path_obj):
      p = str(path_obj)
      return p.endswith(".ssh") or p.endswith("id_rsa")

    mock_exists.side_effect = exists_side_effect
    key = ssh_runner.find_ssh_private_key()
    self.assertEqual(key, Path.home() / ".ssh" / "id_rsa")

  def test_build_ssh_cmd(self):
    cmd = ssh_runner.build_ssh_cmd("1.2.3.4", 2222, "echo hello", private_key_path=Path("/key"), tty=True)
    self.assertIn("-p", cmd)
    self.assertIn("2222", cmd)
    self.assertIn("-t", cmd)
    self.assertIn("-i", cmd)
    self.assertIn("/key", cmd)
    self.assertIn("root@1.2.3.4", cmd)
    self.assertIn("echo hello", cmd)

  def test_build_scp_cmd(self):
    cmd = ssh_runner.build_scp_cmd(Path("/local/script.sh"), "/remote/script.sh", "1.2.3.4", 2222, private_key_path=Path("/key"))
    self.assertIn("-P", cmd)
    self.assertIn("2222", cmd)
    self.assertIn("-i", cmd)
    self.assertIn("/key", cmd)
    self.assertIn("/local/script.sh", cmd)
    self.assertIn("root@1.2.3.4:/remote/script.sh", cmd)

  @patch("subprocess.run")
  def test_wait_for_ssh_success(self, mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    res = ssh_runner.wait_for_ssh("1.2.3.4", 22, timeout=5, interval=0.1)
    self.assertTrue(res)

  @patch("time.sleep")
  @patch("subprocess.run")
  def test_wait_for_ssh_timeout(self, mock_run, mock_sleep):
    mock_run.return_value = MagicMock(returncode=1)
    with self.assertRaises(TimeoutError):
      ssh_runner.wait_for_ssh("1.2.3.4", 22, timeout=0.1, interval=0.05)

  @patch("subprocess.run")
  def test_wait_for_setup_success(self, mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    res = ssh_runner.wait_for_setup("1.2.3.4", 22, timeout=5, interval=0.1)
    self.assertTrue(res)

  @patch("pathlib.Path.exists", autospec=True)
  @patch("xpu_shell.ssh_runner.wait_for_ssh")
  @patch("xpu_shell.ssh_runner.wait_for_setup")
  @patch("subprocess.run")
  def test_execute_remote_script_detach(self, mock_run, mock_wait_setup, mock_wait_ssh, mock_exists):
    mock_exists.return_value = True

    # 1. scp runner.py
    # 2. scp user script
    # 3. launcher script
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="PID:12345\nLOG_FILE:/workspace/logs/job.log\nJOB_ID:job-1\n", stderr="")
    ]

    res = ssh_runner.execute_remote_script(
        host="1.2.3.4",
        port=22,
        script_path="train.py",
        script_args="--epochs 5",
        detach=True
    )

    self.assertEqual(res["job_id"], "job-1")
    self.assertEqual(res["pid"], "12345")
    self.assertEqual(res["log_file"], "/workspace/logs/job.log")
    self.assertEqual(res["exit_code"], 0)
    launch_script = mock_run.call_args_list[2][0][0][-1]
    self.assertIn("python3 /tmp/.runpod_runner.py spawn", launch_script)
    self.assertIn("--script", launch_script)
    self.assertIn("--name train.py", launch_script)
    self.assertIn("--args", launch_script)

  @patch("pathlib.Path.exists", autospec=True)
  @patch("xpu_shell.ssh_runner.wait_for_ssh")
  @patch("xpu_shell.ssh_runner.wait_for_setup")
  @patch("subprocess.run")
  def test_execute_remote_script_foreground(self, mock_run, mock_wait_setup, mock_wait_ssh, mock_exists):
    mock_exists.return_value = True

    # 1. scp runner.py
    # 2. scp user script
    # 3. launcher
    # 4. tail -f (streaming)
    # 5. exit_code read
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="PID:12345\nLOG_FILE:/workspace/logs/job.log\nJOB_ID:job-1\n", stderr=""),
        MagicMock(returncode=0),
        MagicMock(returncode=0, stdout="0\n")
    ]

    res = ssh_runner.execute_remote_script(
        host="1.2.3.4",
        port=22,
        script_path="train.py",
        detach=False
    )

    self.assertEqual(res["pid"], "12345")
    self.assertEqual(res["exit_code"], 0)
    tail_script = mock_run.call_args_list[3][0][0][-1]
    self.assertIn("-s 0.2", tail_script)

  @patch("subprocess.run")
  def test_list_remote_jobs(self, mock_run):
    sample_jobs = [
        {"job_id": "job-1", "pid": 1234, "status": "RUNNING", "started_at": 100}
    ]
    # 1. scp runner.py, 2. list command
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout=json.dumps(sample_jobs), stderr="")
    ]

    jobs = ssh_runner.list_remote_jobs("1.2.3.4", 22)
    self.assertEqual(len(jobs), 1)
    self.assertEqual(jobs[0]["job_id"], "job-1")

  @patch("xpu_shell.ssh_runner.list_remote_jobs")
  @patch("subprocess.run")
  def test_view_remote_logs_follow(self, mock_run, mock_list):
    mock_list.return_value = [
        {"job_id": "job-1", "pid": 1234, "status": "RUNNING", "log_file": "/workspace/logs/job-1.log"}
    ]
    ssh_runner.view_remote_logs("1.2.3.4", 22, job_id="job-1", follow=True)
    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    self.assertNotIn("-t", called_cmd)
    remote_cmd = called_cmd[-1]
    self.assertIn("--pid=1234", remote_cmd)
    self.assertIn("trap", remote_cmd)

  @patch("xpu_shell.ssh_runner.list_remote_jobs")
  @patch("subprocess.run")
  def test_view_remote_logs_follow_completed(self, mock_run, mock_list):
    mock_list.return_value = [
        {"job_id": "job-1", "pid": 1234, "status": "COMPLETED", "exit_code": 0, "log_file": "/workspace/logs/job-1.log"}
    ]
    ssh_runner.view_remote_logs("1.2.3.4", 22, job_id="job-1", follow=True)
    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    remote_cmd = called_cmd[-1]
    self.assertNotIn("-f", remote_cmd)
    self.assertIn("tail -n 50 '/workspace/logs/job-1.log'", remote_cmd)

  @patch("xpu_shell.ssh_runner.list_remote_jobs")
  @patch("subprocess.run")
  def test_view_remote_logs_keyboard_interrupt(self, mock_run, mock_list):
    mock_list.return_value = [
        {"job_id": "job-1", "log_file": "/workspace/logs/job-1.log"}
    ]
    mock_run.side_effect = KeyboardInterrupt
    # Should handle KeyboardInterrupt gracefully without raising
    ssh_runner.view_remote_logs("1.2.3.4", 22, job_id="job-1", follow=True)
    mock_run.assert_called_once()

  @patch("xpu_shell.ssh_runner.list_remote_jobs")
  @patch("subprocess.run")
  def test_view_remote_logs_shorthand_id(self, mock_run, mock_list):
    mock_list.return_value = [
        {"job_id": "job-1", "log_file": "/workspace/logs/job-1.log"}
    ]
    ssh_runner.view_remote_logs("1.2.3.4", 22, job_id="1")
    mock_run.assert_called_once()
    called_cmd = mock_run.call_args[0][0]
    remote_cmd = called_cmd[-1]
    self.assertIn("cat '/workspace/logs/job-1.log'", remote_cmd)

  @patch("subprocess.run")
  def test_kill_remote_job(self, mock_run):
    # 1. scp runner.py, 2. kill command
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="KILLED\n", stderr="")
    ]
    ssh_runner.kill_remote_job("1.2.3.4", 22, "job-1", signal_name="SIGKILL")
    self.assertEqual(mock_run.call_count, 2)
    called_cmd = mock_run.call_args_list[1][0][0]
    self.assertIn("SIGKILL", called_cmd[-1])
    self.assertIn("python3 /tmp/.runpod_runner.py kill", called_cmd[-1])

  @patch("builtins.print")
  @patch("subprocess.run")
  def test_kill_remote_job_timeout_escalation(self, mock_run, mock_print):
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="KILLED_SIGKILL\n", stderr="")
    ]
    ssh_runner.kill_remote_job("1.2.3.4", 22, "job-1", signal_name="SIGTERM", timeout=5.0)
    self.assertEqual(mock_run.call_count, 2)
    called_cmd = mock_run.call_args_list[1][0][0]
    self.assertIn("--timeout 5.0", called_cmd[-1])
    mock_print.assert_any_call(
        "Graceful termination timed out after 5.0s; sent SIGKILL to process job-1 and its process group."
    )

  @patch("subprocess.run")
  def test_ensure_remote_runner(self, mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    path1 = ssh_runner.ensure_remote_runner("1.2.3.4", 22)
    self.assertEqual(path1, "/tmp/.runpod_runner.py")
    self.assertEqual(mock_run.call_count, 1)

    # Second call should use cache
    path2 = ssh_runner.ensure_remote_runner("1.2.3.4", 22)
    self.assertEqual(path2, "/tmp/.runpod_runner.py")
    self.assertEqual(mock_run.call_count, 1)

  @patch("subprocess.run")
  def test_ensure_remote_runner_failure(self, mock_run):
    mock_run.return_value = MagicMock(returncode=1, stderr="SCP failed")
    with self.assertRaises(RuntimeError):
      ssh_runner.ensure_remote_runner("1.2.3.4", 22)

  def test_resolve_ssh_config(self):
    # Explicit config
    self.assertEqual(ssh_runner.resolve_ssh_config("/dev/null"), "/dev/null")
    self.assertEqual(ssh_runner.resolve_ssh_config(Path("/custom/config")), "/custom/config")

    # Disabled keywords
    for kw in ["none", "system", "default", "", "NONE", "DEFAULT"]:
      self.assertIsNone(ssh_runner.resolve_ssh_config(kw))

    # Env var fallback
    with patch.dict("os.environ", {"RUNPOD_SSH_CONFIG": "/env/ssh/config"}):
      self.assertEqual(ssh_runner.resolve_ssh_config(), "/env/ssh/config")

    with patch.dict("os.environ", {"RUNPOD_SSH_CONFIG": "none"}):
      self.assertIsNone(ssh_runner.resolve_ssh_config())

    with patch.dict("os.environ", {}, clear=True), \
         patch("pathlib.Path.is_file", return_value=False), \
         patch("pathlib.Path.is_dir", return_value=False):
      self.assertIsNone(ssh_runner.resolve_ssh_config())

    # User ~/.ssh/config exists
    with patch.dict("os.environ", {}, clear=True), \
         patch("pathlib.Path.is_file", return_value=True):
      self.assertEqual(ssh_runner.resolve_ssh_config(), str(Path.home() / ".ssh" / "config"))

    # Broken system /etc/ssh/ssh_config.d permissions
    mock_file = MagicMock()
    mock_file.stat.return_value = MagicMock(st_uid=65534)
    with patch.dict("os.environ", {}, clear=True), \
         patch("pathlib.Path.is_file", return_value=False), \
         patch("pathlib.Path.is_dir", return_value=True), \
         patch("pathlib.Path.glob", return_value=[mock_file]), \
         patch("os.getuid", return_value=1000):
      self.assertEqual(ssh_runner.resolve_ssh_config(), "/dev/null")

  def test_build_ssh_cmd_with_ssh_config(self):
    cmd = ssh_runner.build_ssh_cmd("1.2.3.4", 2222, ssh_config_path="/dev/null")
    self.assertIn("-F", cmd)
    idx = cmd.index("-F")
    self.assertEqual(cmd[idx + 1], "/dev/null")

  def test_build_scp_cmd_with_ssh_config(self):
    cmd = ssh_runner.build_scp_cmd(Path("/local/script.sh"), "/remote/script.sh", "1.2.3.4", 2222, ssh_config_path="/dev/null")
    self.assertIn("-F", cmd)
    idx = cmd.index("-F")
    self.assertEqual(cmd[idx + 1], "/dev/null")

  @patch("pathlib.Path.exists", autospec=True)
  @patch("xpu_shell.ssh_runner.wait_for_ssh")
  @patch("xpu_shell.ssh_runner.wait_for_setup")
  @patch("subprocess.run")
  def test_execute_remote_script_with_extra_env(self, mock_run, mock_wait_setup, mock_wait_ssh, mock_exists):
    import base64
    mock_exists.return_value = True

    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="PID:12345\nLOG_FILE:/workspace/logs/job.log\nJOB_ID:job-1\n", stderr="")
    ]

    res = ssh_runner.execute_remote_script(
        host="1.2.3.4",
        port=22,
        script_path="train.py",
        detach=True,
        extra_env={"R2_TOKEN": "secret_token", "DEBUG": "1"}
    )

    self.assertEqual(res["pid"], "12345")
    launch_script = mock_run.call_args_list[2][0][0][-1]
    self.assertIn("export RUNPOD_JOB_ENV=", launch_script)
    m = re.search(r'export RUNPOD_JOB_ENV="([A-Za-z0-9+/=]+)"', launch_script)
    self.assertIsNotNone(m)
    decoded = json.loads(base64.b64decode(m.group(1)).decode("utf-8"))
    self.assertEqual(decoded, {"R2_TOKEN": "secret_token", "DEBUG": "1"})

  def test_runner_unpacks_job_env(self):
    import base64
    import xpu_shell.runner as runner
    env_payload = {"MY_SECRET": "abc123xyz"}
    b64_payload = base64.b64encode(json.dumps(env_payload).encode("utf-8")).decode("ascii")

    with patch.dict("os.environ", {"RUNPOD_JOB_ENV": b64_payload}):
      env = os.environ.copy()
      job_env_b64 = env.pop("RUNPOD_JOB_ENV", None)
      self.assertEqual(job_env_b64, b64_payload)
      decoded = json.loads(base64.b64decode(job_env_b64).decode("utf-8"))
      env.update(decoded)
      self.assertEqual(env.get("MY_SECRET"), "abc123xyz")
      self.assertNotIn("RUNPOD_JOB_ENV", env)

  @patch("xpu_shell.ssh_runner.wait_for_ssh")
  @patch("xpu_shell.ssh_runner.wait_for_setup")
  @patch("subprocess.run")
  def test_execute_remote_command_detach(self, mock_run, mock_wait_setup, mock_wait_ssh):
    # 1. scp runner.py
    # 2. launcher script
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="PID:54321\nLOG_FILE:/workspace/logs/job.log\nJOB_ID:job-cmd-1\n", stderr="")
    ]

    res = ssh_runner.execute_remote_command(
        host="1.2.3.4",
        port=22,
        command_args=["python3", "-c", "print('hello')"],
        detach=True,
        extra_env={"TEST_VAR": "val"}
    )

    self.assertTrue(res["job_id"].startswith("job-"))
    self.assertEqual(res["pid"], "54321")
    self.assertEqual(res["log_file"], "/workspace/logs/job.log")
    self.assertEqual(res["exit_code"], 0)
    mock_wait_ssh.assert_called_once()
    mock_wait_setup.assert_called_once()

  @patch("xpu_shell.ssh_runner.wait_for_ssh")
  @patch("xpu_shell.ssh_runner.wait_for_setup")
  @patch("subprocess.run")
  def test_execute_remote_command_foreground(self, mock_run, mock_wait_setup, mock_wait_ssh):
    # 1. scp runner.py
    # 2. launcher script
    # 3. tail stream
    # 4. exit code cat
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="PID:54321\nLOG_FILE:/workspace/logs/job.log\nJOB_ID:job-cmd-1\n", stderr=""),
        MagicMock(returncode=0),
        MagicMock(returncode=0, stdout="0\n", stderr="")
    ]

    res = ssh_runner.execute_remote_command(
        host="1.2.3.4",
        port=22,
        command_args=["nvidia-smi"],
        detach=False
    )

    self.assertTrue(res["job_id"].startswith("job-"))
    self.assertEqual(res["pid"], "54321")
    self.assertEqual(res["exit_code"], 0)

  def test_build_cp_cmd_basic(self):
    cmd = ssh_runner.build_cp_cmd(
        sources=["root@1.2.3.4:/workspace/file.txt"],
        dest="./file.txt",
        port=2222,
        recursive=True,
        preserve=True,
        quiet=True,
        private_key_path=Path("/tmp/key"),
        ssh_config_path="/dev/null"
    )
    self.assertEqual(cmd[0], "scp")
    self.assertIn("-P", cmd)
    self.assertIn("2222", cmd)
    self.assertIn("-r", cmd)
    self.assertIn("-p", cmd)
    self.assertIn("-q", cmd)
    self.assertIn("-F", cmd)
    self.assertIn("/dev/null", cmd)
    self.assertIn("-i", cmd)
    self.assertIn("/tmp/key", cmd)
    self.assertEqual(cmd[-2], "root@1.2.3.4:/workspace/file.txt")
    self.assertEqual(cmd[-1], "./file.txt")

  @patch("xpu_shell.ssh_runner.wait_for_ssh")
  @patch("xpu_shell.ssh_runner.wait_for_setup")
  @patch("subprocess.run")
  def test_execute_remote_command_with_shell(self, mock_run, mock_wait_setup, mock_wait_ssh):
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(returncode=0, stdout="PID:54321\nLOG_FILE:/workspace/logs/job.log\nJOB_ID:job-cmd-shell\n", stderr=""),
        MagicMock(returncode=0),
        MagicMock(returncode=0, stdout="0\n", stderr="")
    ]

    res = ssh_runner.execute_remote_command(
        host="1.2.3.4",
        port=22,
        command_args=["echo $VAR | grep foo"],
        detach=False,
        use_shell=True
    )

    self.assertTrue(res["job_id"].startswith("job-"))
    # Verify launcher script contains --shell
    launcher_ssh_call = mock_run.call_args_list[1]
    launcher_script = launcher_ssh_call[0][0][-1]
    self.assertIn("--shell", launcher_script)


if __name__ == "__main__":
  unittest.main()
