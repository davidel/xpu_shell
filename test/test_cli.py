import sys
from unittest.mock import MagicMock
# Mock runpod module to avoid ModuleNotFoundError when importing cli
sys.modules['runpod'] = MagicMock()

import base64
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch, mock_open

# Import the module under test
import xpu_shell.cli as cli


class TestRunPodShellCLI(unittest.TestCase):

  def setUp(self):
    self._temp_dir = tempfile.TemporaryDirectory()
    self._env_patcher = patch.dict(os.environ, {
        "XPU_SHELL_CONFIG_DIR": self._temp_dir.name,
        "RUNPOD_SHELL_CONFIG_DIR": self._temp_dir.name
    })
    self._env_patcher.start()

  def tearDown(self):
    self._env_patcher.stop()
    self._temp_dir.cleanup()

  @patch("pathlib.Path.exists", autospec=True)
  @patch("builtins.open", new_callable=mock_open, read_data="ssh-rsa AAAAB3...")
  def test_get_ssh_key_explicit(self, mock_file, mock_exists):
    mock_exists.return_value = True
    key = cli.get_ssh_key("~/.ssh/id_ed25519.pub")
    self.assertEqual(key, "ssh-rsa AAAAB3...")
    mock_file.assert_called_once_with(Path("~/.ssh/id_ed25519.pub").expanduser(), 'r')

  def test_get_ssh_key_explicit_not_found(self):
    with patch("pathlib.Path.exists", autospec=True, return_value=False):
      with self.assertRaises(FileNotFoundError):
        cli.get_ssh_key("~/.ssh/nonexistent.pub")

  @patch("pathlib.Path.exists", autospec=True)
  @patch("builtins.open", new_callable=mock_open, read_data="ssh-rsa default_key...")
  def test_get_ssh_key_default(self, mock_file, mock_exists):
    def exists_side_effect(path_obj):
      path_str = str(path_obj)
      if ".ssh" in path_str:
        if path_str.endswith("id_rsa.pub") or path_str.endswith(".ssh"):
          return True
      return False

    mock_exists.side_effect = exists_side_effect

    key = cli.get_ssh_key()
    self.assertEqual(key, "ssh-rsa default_key...")

  @patch("pathlib.Path.exists", autospec=True)
  @patch("builtins.open", new_callable=mock_open, read_data="numpy==1.24.3")
  def test_read_requirements_exists(self, mock_file, mock_exists):
    mock_exists.return_value = True
    content = cli.read_requirements("requirements.txt")
    self.assertEqual(content, "numpy==1.24.3")

  @patch("pathlib.Path.exists", autospec=True)
  def test_read_requirements_not_exists_default(self, mock_exists):
    mock_exists.return_value = False
    content = cli.read_requirements()
    self.assertEqual(content, "")

  @patch("pathlib.Path.exists", autospec=True)
  @patch("builtins.open", new_callable=mock_open, read_data="# Comment\nKEY=VALUE\nOTHER='quotes'\n")
  def test_parse_env_file(self, mock_file, mock_exists):
    mock_exists.return_value = True
    env_vars = cli.parse_env_file(".env")
    self.assertEqual(env_vars, {"KEY": "VALUE", "OTHER": "quotes"})

  @patch("pathlib.Path.exists", autospec=True)
  @patch("builtins.open", new_callable=mock_open, read_data="# Comment\nscreen curl\nhtop\n")
  def test_read_apt_packages_file(self, mock_file, mock_exists):
    mock_exists.return_value = True
    packages = cli.read_apt_packages_file("apt.txt")
    self.assertEqual(packages, ["screen", "curl", "htop"])

  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.run_container_setup")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_command_escaping(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_setup, mock_wait_ssh):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = "importlib-metadata==6.7.0; python_version < '3.8'"
    mock_apt_file.return_value = ["git"]

    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [
                {
                    "privatePort": 22,
                    "isExternal": 12345,
                    "address": "12.34.56.78"
                }
            ]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-worker",
        "--image-name", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--pip-packages", "scipy"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]

    self.assertEqual(kwargs["name"], "test-worker")
    self.assertEqual(kwargs["gpu_count"], 1)
    self.assertEqual(kwargs["min_vcpu_count"], 4)
    self.assertEqual(kwargs["min_memory_in_gb"], 8)
    self.assertEqual(kwargs["volume_mount_path"], "/workspace")
    self.assertNotIn("docker_args", kwargs)

    mock_wait_ssh.assert_called_once()
    mock_setup.assert_called_once()
    setup_kwargs = mock_setup.call_args[1]
    setup_script = setup_kwargs["setup_script_content"]
    self.assertIn("echo 'importlib-metadata==6.7.0; python_version < '\\''3.8'\\''' > \"/workspace/requirements.txt\"", setup_script)
    self.assertIn("break-system-packages = true", setup_script)
    self.assertIn('pip install -r "/workspace/requirements.txt"', setup_script)
    self.assertIn("pip install scipy", setup_script)
    self.assertNotIn("venv", setup_script.lower())

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_command_custom_vcpu_and_memory(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []

    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [
                {
                    "privatePort": 22,
                    "isExternal": 12345,
                    "address": "12.34.56.78"
                }
            ]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-worker-custom",
        "--image-name", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--vcpu-count", "8",
        "--memory", "16"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]

    self.assertEqual(kwargs["name"], "test-worker-custom")
    self.assertEqual(kwargs["min_vcpu_count"], 8)
    self.assertEqual(kwargs["min_memory_in_gb"], 16)
    self.assertNotIn("docker_args", kwargs)
    mock_wait_ssh.assert_called_once()

  @patch("sys.stderr")
  def test_fatal(self, mock_stderr):
    with self.assertRaises(ValueError):
      cli.fatal("An error occurred", exc=ValueError)
    mock_stderr.write.assert_any_call("An error occurred")

  @patch("runpod.get_gpus")
  def test_get_valid_gpus(self, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090"},
        {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000"}
    ]
    gpus = cli.get_valid_gpus()
    self.assertEqual(gpus, ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"])

  def test_resolve_gpu_type(self):
    valid = ["NVIDIA GeForce RTX 4090", "NVIDIA RTX A6000"]
    # Exact match
    self.assertEqual(cli.resolve_gpu_type("NVIDIA GeForce RTX 4090", valid), "NVIDIA GeForce RTX 4090")
    # Case-insensitive
    self.assertEqual(cli.resolve_gpu_type("nvidia geforce rtx 4090", valid), "NVIDIA GeForce RTX 4090")
    # Substring match (unique)
    self.assertEqual(cli.resolve_gpu_type("4090", valid), "NVIDIA GeForce RTX 4090")
    # Substring match (ambiguous)
    with self.assertRaises(ValueError):
      cli.resolve_gpu_type("RTX", valid)
    # Not found / Fuzzy match close suggestions
    with self.assertRaises(ValueError):
      cli.resolve_gpu_type("NVIDA 4090", valid)

  @patch("runpod.get_gpus")
  @patch("sys.stdout")
  def test_gpus_command(self, mock_stdout, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24}
    ]
    test_args = ["cli.py", "gpus"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_stdout.write.assert_any_call("RTX 4090                       | RTX 4090                  | 24        | N/A | N/A     | N/A      ")

  @patch("runpod.get_gpus")
  @patch("sys.stdout")
  def test_gpus_command_regex_match_id(self, mock_stdout, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24},
        {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "memoryInGb": 48}
    ]
    test_args = ["cli.py", "gpus", "4090"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    calls = [call[0][0] for call in mock_stdout.write.call_args_list]
    matched_lines = [line for line in calls if "4090" in line]
    unmatched_lines = [line for line in calls if "A6000" in line]
    self.assertTrue(len(matched_lines) > 0)
    self.assertEqual(len(unmatched_lines), 0)

  @patch("runpod.get_gpus")
  @patch("sys.stdout")
  def test_gpus_command_regex_match_display_name(self, mock_stdout, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "AMD Instinct MI300X OAM", "displayName": "MI300X", "memoryInGb": 192},
        {"id": "NVIDIA RTX A6000", "displayName": "RTX A6000", "memoryInGb": 48}
    ]
    test_args = ["cli.py", "gpus", "-r", "mi300"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    calls = [call[0][0] for call in mock_stdout.write.call_args_list]
    matched_lines = [line for line in calls if "MI300X" in line]
    unmatched_lines = [line for line in calls if "A6000" in line]
    self.assertTrue(len(matched_lines) > 0)
    self.assertEqual(len(unmatched_lines), 0)

  @patch("runpod.get_gpus")
  @patch("sys.stdout")
  def test_gpus_command_regex_no_match(self, mock_stdout, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24}
    ]
    test_args = ["cli.py", "gpus", "nonexistent"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_stdout.write.assert_any_call("No GPUs matching pattern 'nonexistent' found.")

  @patch("runpod.get_gpus")
  def test_gpus_command_invalid_regex(self, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24}
    ]
    test_args = ["cli.py", "gpus", "["]
    with patch.object(sys, "argv", test_args):
      with self.assertRaises(ValueError):
        cli.main()

  @patch("runpod.get_gpus")
  @patch("sys.stdout")
  def test_gpus_command_expand_runpod(self, mock_stdout, mock_get_gpus):
    mock_get_gpus.return_value = [
        {"id": "RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24}
    ]
    test_args = ["cli.py", "gpus", "-l"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("Note: RunPod allocates pooled GPU capacity", output)
    self.assertIn("RTX 4090", output)

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_with_run_script(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_find_priv, mock_exec, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_find_priv.return_value = Path("/fake/priv_key")
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-worker",
        "--image-name", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--run-script", "script.sh",
        "--script-args", "--flag 1"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_wait_ssh.assert_called_once()
    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="script.sh",
        script_args="--flag 1",
        detach=False,
        private_key_path=Path("/fake/priv_key"),
        wait_for_setup_flag=False,
        ssh_timeout=180,
        ssh_config_path=None
    )

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_with_ssh_config(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_find_priv, mock_exec, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_find_priv.return_value = Path("/fake/priv_key")
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-worker",
        "--image-name", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--run-script", "script.sh",
        "--ssh-config", "/dev/null"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_wait_ssh.assert_called_once()
    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="script.sh",
        script_args="",
        detach=False,
        private_key_path=Path("/fake/priv_key"),
        wait_for_setup_flag=False,
        ssh_timeout=180,
        ssh_config_path="/dev/null"
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_command(self, mock_get_pod, mock_exec, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "pod-123",
        "script.sh",
        "--script-args", "arg1 arg2",
        "--detach",
        "--ssh-config", "/dev/null"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="script.sh",
        script_args="arg1 arg2",
        detach=True,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path="/dev/null",
        extra_env=None,
        verbose=False
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.list_remote_jobs")
  @patch("runpod.get_pod")
  @patch("sys.stdout")
  def test_ps_command(self, mock_stdout, mock_get_pod, mock_list_jobs, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_list_jobs.return_value = [
        {
            "job_id": "job-1",
            "pid": 1234,
            "status": "RUNNING",
            "started_at_iso": "2026-09-04T12:00:00Z",
            "duration": "5m 0s",
            "script": "train.py",
            "log_file": "/workspace/logs/job-1.log"
        }
    ]

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "ps",
        "pod-123",
        "--ssh-config", "/dev/null"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_list_jobs.assert_called_once_with("12.34.56.78", 12345, private_key_path=None, ssh_config_path="/dev/null")

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.list_remote_jobs")
  @patch("runpod.get_pod")
  @patch("builtins.print")
  def test_ps_table_formatting_compact(self, mock_print, mock_get_pod, mock_list_jobs, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_list_jobs.return_value = [
        {
            "job_id": "job-1",
            "pid": 1234,
            "status": "RUNNING",
            "started_at_iso": "2026-09-04T12:00:00Z",
            "duration": "5m 0s",
            "script": "train.py",
            "log_file": "/workspace/logs/job-1.log"
        }
    ]

    test_args = ["cli.py", "--api-key", "fake-api-key", "ps", "pod-123"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    printed_lines = [call[0][0] for call in mock_print.call_args_list if call[0]]
    header_line = next((l for l in printed_lines if "JOB ID" in l), None)
    self.assertIsNotNone(header_line)
    self.assertTrue(header_line.startswith("JOB ID   |"))

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.view_remote_logs")
  @patch("runpod.get_pod")
  def test_logs_command(self, mock_get_pod, mock_view_logs, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "logs",
        "pod-123",
        "job-1",
        "-f",
        "--ssh-config", "/dev/null"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_view_logs.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        job_id="job-1",
        tail_lines=None,
        follow=True,
        private_key_path=None,
        ssh_config_path="/dev/null",
        verbose=False
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.kill_remote_job")
  @patch("runpod.get_pod")
  def test_kill_command(self, mock_get_pod, mock_kill, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "kill",
        "pod-123",
        "job-1",
        "-s", "SIGKILL",
        "--ssh-config", "/dev/null"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_kill.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        target_id="job-1",
        signal_name="SIGKILL",
        timeout=30.0,
        private_key_path=None,
        ssh_config_path="/dev/null"
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.kill_remote_job")
  @patch("runpod.get_pod")
  def test_kill_command_custom_timeout(self, mock_get_pod, mock_kill, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }

    test_args = [
        "runpod-shell",
        "kill",
        "pod-123",
        "job-1",
        "-t", "5.5",
        "--ssh-config", "/dev/null"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_kill.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        target_id="job-1",
        signal_name="SIGTERM",
        timeout=5.5,
        private_key_path=None,
        ssh_config_path="/dev/null"
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_with_env_ssh_config(self, mock_get_pod, mock_exec, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "pod-123",
        "script.sh"
    ]

    with patch.dict("os.environ", {"RUNPOD_SSH_CONFIG": "/env/ssh/config"}):
      with patch.object(sys, "argv", test_args):
        cli.main()

    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="script.sh",
        script_args="",
        detach=False,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path="/env/ssh/config",
        extra_env=None,
        verbose=False
    )

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.get_pod_template")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_with_template_id(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_get_template, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_get_template.return_value = {
        "id": "runpod-torch-v280",
        "name": "Runpod Pytorch 2.8.0",
        "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    }
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-template-worker",
        "--template-id", "runpod-torch-v280"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_wait_ssh.assert_called_once()
    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]
    self.assertEqual(kwargs["name"], "test-template-worker")
    self.assertEqual(kwargs["template_id"], "runpod-torch-v280")
    self.assertEqual(kwargs["image_name"], "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")
    self.assertNotIn("docker_args", kwargs)

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.get_pod_template")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_template_id_precedence_over_image_name(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_get_template, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_get_template.return_value = {
        "id": "runpod-torch-v280",
        "name": "Runpod Pytorch 2.8.0",
        "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    }
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-precedence-worker",
        "--template-id", "runpod-torch-v280",
        "--image-name", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_wait_ssh.assert_called_once()
    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]
    self.assertEqual(kwargs["name"], "test-precedence-worker")
    self.assertEqual(kwargs["template_id"], "runpod-torch-v280")
    self.assertEqual(kwargs["image_name"], "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")

  @patch("xpu_shell.cli.get_pod_template")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  def test_create_template_not_found_fails(self, mock_apt_file, mock_req, mock_ssh_key, mock_get_template):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_get_template.return_value = None

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-fail-worker",
        "--template-id", "invalid-template"
    ]

    with patch.object(sys, "argv", test_args):
      with self.assertRaises(ValueError) as ctx:
        cli.main()
      self.assertIn("Template 'invalid-template' not found", str(ctx.exception))

  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_without_template_or_image_fails(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-fail-worker"
    ]

    with patch.object(sys, "argv", test_args):
      with self.assertRaises(ValueError) as ctx:
        cli.main()
      self.assertIn("Either --template-id or --image-name must be specified.", str(ctx.exception))

  @patch("sys.stdout")
  @patch("xpu_shell.cli.get_pod_templates")
  def test_cmd_templates_listing_and_filtering(self, mock_get_templates, mock_stdout):
    mock_get_templates.return_value = [
        {
            "id": "runpod-torch-v240",
            "name": "Runpod Pytorch 2.4.0",
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
        },
        {
            "id": "runpod-torch-v280",
            "name": "Runpod Pytorch 2.8.0",
            "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
        },
        {
            "id": "runpod-ubuntu",
            "name": "Runpod Ubuntu 20.04",
            "imageName": "runpod/base:0.7.0-ubuntu2004"
        }
    ]

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "templates",
        "--filter", "torch"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_get_templates.assert_called_once()
    calls = [call[0][0] for call in mock_stdout.write.call_args_list]
    matched_lines = [line for line in calls if "runpod-torch" in line]
    unmatched_lines = [line for line in calls if "runpod-ubuntu" in line]
    self.assertTrue(len(matched_lines) > 0)
    self.assertEqual(len(unmatched_lines), 0)

  @patch("xpu_shell.cli.run_graphql")
  def test_get_pod_template_direct(self, mock_graphql):
    mock_graphql.return_value = {
        "data": {
            "podTemplate": {
                "id": "runpod-torch-v280",
                "name": "Runpod Pytorch 2.8.0",
                "imageName": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
            }
        }
    }
    template = cli.get_pod_template("runpod-torch-v280")
    self.assertIsNotNone(template)
    self.assertEqual(template["id"], "runpod-torch-v280")
    self.assertEqual(template["imageName"], "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404")

  @patch("xpu_shell.cli.get_pod_templates")
  @patch("xpu_shell.cli.run_graphql")
  def test_get_pod_template_fallback(self, mock_graphql, mock_get_templates):
    mock_graphql.return_value = {"data": {"podTemplate": None}}
    mock_get_templates.return_value = [
        {
            "id": "custom-template",
            "name": "Custom",
            "imageName": "custom/image:tag"
        }
    ]
    template = cli.get_pod_template("custom-template")
    self.assertIsNotNone(template)
    self.assertEqual(template["id"], "custom-template")
    self.assertEqual(template["imageName"], "custom/image:tag")

  def test_build_container_setup_script_quoting(self):
    script = cli.build_container_setup_script(
        apt_packages=["screen", "tree", "gcc"],
        requirements_content="",
        pip_packages=["scdiag[all] @ git+https://github.com/davidel/scdiag.git", "nvidia-ml-py"],
        volume_mount_path="/workspace"
    )
    self.assertIn("install 'scdiag[all] @ git+https://github.com/davidel/scdiag.git' nvidia-ml-py", script)
    self.assertIn("apt-get install -y screen tree gcc", script)

  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.run_container_setup")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_command_pip_packages_with_spaces(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_setup, mock_wait_ssh):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "gpus": True,
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-worker",
        "--image-name", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
        "--pip-packages", "scdiag[all] @ git+https://github.com/davidel/scdiag.git", "nvidia-ml-py"
    ]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]
    self.assertNotIn("docker_args", kwargs)
    mock_wait_ssh.assert_called_once()
    mock_setup.assert_called_once()
    setup_kwargs = mock_setup.call_args[1]
    setup_script = setup_kwargs["setup_script_content"]
    self.assertIn("install 'scdiag[all] @ git+https://github.com/davidel/scdiag.git' nvidia-ml-py", setup_script)

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_with_explicit_docker_args(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [{"privatePort": 22, "publicPort": 12345, "ip": "12.34.56.78"}]
        }
    }
    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-custom-docker",
        "--image-name", "custom/image:tag",
        "--docker-args", "bash -c 'sleep 3600'"
    ]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]
    self.assertEqual(kwargs.get("docker_args"), "bash -c 'sleep 3600'")

  @patch("subprocess.run")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_with_custom_volume_mount_path(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key, mock_wait_ssh, mock_sub_run):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "RUNNING",
        "runtime": {
            "ports": [{"privatePort": 22, "publicPort": 12345, "ip": "12.34.56.78"}]
        }
    }
    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-custom-mount",
        "--image-name", "custom/image:tag",
        "--volume-size", "150",
        "--volume-mount-path", "/custom/mount"
    ]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_create_pod.assert_called_once()
    kwargs = mock_create_pod.call_args[1]
    self.assertEqual(kwargs.get("volume_in_gb"), 150)
    self.assertEqual(kwargs.get("volume_mount_path"), "/custom/mount")

  def test_get_pod_ssh_endpoint_graphql_fields(self):
    pod_info = {
        "runtime": {
            "ports": [
                {
                    "ip": "1.2.3.4",
                    "privatePort": 22,
                    "publicPort": 54321
                }
            ]
        }
    }
    host, port = cli.get_pod_ssh_endpoint(pod_info)
    self.assertEqual(host, "1.2.3.4")
    self.assertEqual(port, 54321)

  @patch("runpod.get_pods")
  @patch("sys.stdout")
  def test_cmd_list_with_graphql_endpoint(self, mock_stdout, mock_get_pods):
    mock_get_pods.return_value = [
        {
            "id": "pod-123",
            "name": "my-worker",
            "desiredStatus": "RUNNING",
            "machine": {
                "gpuDisplayName": "RTX 4090"
            },
            "gpuCount": 1,
            "runtime": {
                "ports": [
                    {
                        "ip": "1.2.3.4",
                        "privatePort": 22,
                        "publicPort": 54321
                    }
                ]
            }
        }
    ]
    test_args = ["cli.py", "--api-key", "fake-api-key", "list"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("1.2.3.4:54321", output)
    self.assertIn("1x RTX 4090", output)

  @patch("xpu_shell.cli.get_ssh_key")
  @patch("xpu_shell.cli.read_requirements")
  @patch("xpu_shell.cli.read_apt_packages_file")
  @patch("runpod.create_pod")
  @patch("runpod.get_pod")
  @patch("time.sleep")
  def test_create_polling_failed_status(self, mock_sleep, mock_get_pod, mock_create_pod, mock_apt_file, mock_req, mock_ssh_key):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    mock_req.return_value = ""
    mock_apt_file.return_value = []
    mock_create_pod.return_value = {"id": "pod-123"}
    mock_get_pod.return_value = {
        "id": "pod-123",
        "desiredStatus": "FAILED"
    }
    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--name", "test-fail-worker",
        "--image-name", "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
    ]
    with patch.object(sys, "argv", test_args):
      with self.assertRaises(RuntimeError) as ctx:
        cli.main()
      self.assertIn("Pod initialization failed with status: FAILED", str(ctx.exception))

  @patch("xpu_shell.cli.execute_remote_script")
  def test_run_container_setup_failure(self, mock_exec):
    mock_exec.return_value = {"exit_code": 1}
    with self.assertRaises(RuntimeError) as ctx:
      cli.run_container_setup("1.2.3.4", 22, "#!/bin/bash\nexit 1")
    self.assertIn("Container setup failed with exit code: 1", str(ctx.exception))

  def test_parse_env_explicit(self):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", nargs="+", action=cli.ParseEnv, default={})
    args = parser.parse_args(["-e", "FOO=bar", "BAZ=qux"])
    self.assertEqual(args.env, {"FOO": "bar", "BAZ": "qux"})

  def test_parse_env_from_local_environ(self):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", nargs="+", action=cli.ParseEnv, default={})
    with patch.dict("os.environ", {"MY_VAR": "secret_val"}):
      args = parser.parse_args(["-e", "MY_VAR", "ANOTHER=123"])
      self.assertEqual(args.env, {"MY_VAR": "secret_val", "ANOTHER": "123"})

  def test_parse_env_missing_local_environ(self):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", nargs="+", action=cli.ParseEnv, default={})
    with patch.dict("os.environ", {}, clear=True):
      with self.assertRaises(SystemExit):
        with patch("sys.stderr"):
          parser.parse_args(["-e", "NONEXISTENT_VAR"])

  @patch("pathlib.Path.exists", autospec=True)
  @patch("builtins.open", new_callable=mock_open, read_data="export API_KEY=\"secret\"\nDEBUG=true\n")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_with_env_and_env_file(self, mock_get_pod, mock_exec, mock_find_priv, mock_file, mock_exists):
    mock_exists.return_value = True
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "publicPort": 12345, "ip": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "pod-123",
        "script.sh",
        "--env-file", ".env",
        "-e", "LOCAL_VAR", "OVERRIDE=custom"
    ]

    with patch.dict("os.environ", {"LOCAL_VAR": "local_value"}):
      with patch.object(sys, "argv", test_args):
        cli.main()

    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="script.sh",
        script_args="",
        detach=False,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env={
            "API_KEY": "secret",
            "DEBUG": "true",
            "LOCAL_VAR": "local_value",
            "OVERRIDE": "custom"
        },
        verbose=False
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_env_with_positional_pod_and_script(self, mock_get_pod, mock_exec, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "-d",
        "-e", "R2_TOKEN", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "CLOUDFLARE_ACCOUNT_ID",
        "pod-123",
        "train.py"
    ]

    mock_env = {
        "R2_TOKEN": "token-xyz",
        "R2_ACCESS_KEY_ID": "access-123",
        "R2_SECRET_ACCESS_KEY": "secret-456",
        "CLOUDFLARE_ACCOUNT_ID": "cf-789"
    }

    with patch.dict("os.environ", mock_env):
      with patch.object(sys, "argv", test_args):
        cli.main()

    mock_get_pod.assert_called_once_with("pod-123")
    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="train.py",
        script_args="",
        detach=True,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env=mock_env,
        verbose=False
    )

  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_env_with_last_pod_id(self, mock_get_pod, mock_exec, mock_find_priv, mock_last_pod):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "-d",
        "-e", "R2_TOKEN", "R2_ACCESS_KEY_ID",
        "train.py"
    ]

    mock_env = {
        "R2_TOKEN": "token-xyz",
        "R2_ACCESS_KEY_ID": "access-123"
    }

    with patch.dict("os.environ", mock_env):
      with patch.object(sys, "argv", test_args):
        cli.main()

    mock_get_pod.assert_called_once_with("pod-last")
    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="train.py",
        script_args="",
        detach=True,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env=mock_env,
        verbose=False
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_with_pod_flag(self, mock_get_pod, mock_exec, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-custom",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "--pod", "pod-custom",
        "-d",
        "-e", "R2_TOKEN",
        "train.py"
    ]

    mock_env = {"R2_TOKEN": "token-xyz"}

    with patch.dict("os.environ", mock_env):
      with patch.object(sys, "argv", test_args):
        cli.main()

    mock_get_pod.assert_called_once_with("pod-custom")
    mock_exec.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        script_path="train.py",
        script_args="",
        detach=True,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env=mock_env,
        verbose=False
    )

  @patch("runpod.stop_pod")
  @patch("xpu_shell.cli.get_last_pod_id")
  def test_stop_with_last_pod_id(self, mock_last_pod, mock_stop):
    mock_last_pod.return_value = "pod-last"
    test_args = ["cli.py", "--api-key", "fake-api-key", "stop"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_stop.assert_called_once_with("pod-last")

  @patch("runpod.stop_pod")
  def test_stop_with_pod_flag(self, mock_stop):
    test_args = ["cli.py", "--api-key", "fake-api-key", "stop", "--pod", "pod-flag"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_stop.assert_called_once_with("pod-flag")

  @patch("runpod.terminate_pod")
  @patch("xpu_shell.cli.clear_last_pod_id_if_match")
  @patch("xpu_shell.cli.get_last_pod_id")
  def test_terminate_with_last_pod_id(self, mock_last_pod, mock_clear, mock_term):
    mock_last_pod.return_value = "pod-last"
    test_args = ["cli.py", "--api-key", "fake-api-key", "terminate"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_term.assert_called_once_with("pod-last")
    mock_clear.assert_called_once_with("pod-last")

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.list_remote_jobs")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_ps_with_last_pod_id(self, mock_get_pod, mock_last_pod, mock_list_jobs, mock_find_priv):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_list_jobs.return_value = []
    test_args = ["cli.py", "--api-key", "fake-api-key", "ps"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.kill_remote_job")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_kill_with_last_pod_id(self, mock_get_pod, mock_last_pod, mock_kill, mock_find_priv):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = ["cli.py", "--api-key", "fake-api-key", "kill", "job-99"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")
    mock_kill.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        target_id="job-99",
        signal_name="SIGTERM",
        timeout=30.0,
        private_key_path=None,
        ssh_config_path=None
    )

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.view_remote_logs")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_logs_with_last_pod_id(self, mock_get_pod, mock_last_pod, mock_logs, mock_find_priv):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = ["cli.py", "--api-key", "fake-api-key", "logs", "job-1"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")
    mock_logs.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        job_id="job-1",
        tail_lines=None,
        follow=False,
        private_key_path=None,
        ssh_config_path=None,
        verbose=False
    )

  def test_save_and_get_last_pod_id(self):
    with tempfile.TemporaryDirectory() as td:
      fake_file = Path(td) / ".last_pod_id"
      with patch.dict(os.environ, {"XPU_SHELL_LAST_POD_ID_FILE": str(fake_file)}):
        self.assertIsNone(cli.get_last_pod_id())

        cli.save_last_pod_id("pod-abc-123")
        self.assertEqual(cli.get_last_pod_id(), "pod-abc-123")
        cli.clear_last_pod_id_if_match("pod-other")
        self.assertEqual(cli.get_last_pod_id(), "pod-abc-123")
        cli.clear_last_pod_id_if_match("pod-abc-123")
        self.assertIsNone(cli.get_last_pod_id())

  def test_pod_id_file_env_overrides(self):
    with tempfile.TemporaryDirectory() as td:
      dir_path = Path(td) / "custom_config"
      file_path = Path(td) / "custom_file"
      with patch.dict(os.environ, {"XPU_SHELL_CONFIG_DIR": str(dir_path)}, clear=True):
        self.assertEqual(cli.get_config_dir(), dir_path)
        self.assertEqual(cli.get_last_pod_id_file(), dir_path / ".last_pod_id")

      with patch.dict(os.environ, {"XPU_SHELL_LAST_POD_ID_FILE": str(file_path)}, clear=True):
        self.assertEqual(cli.get_last_pod_id_file(), file_path)

      with patch.dict(os.environ, {}, clear=True):
        self.assertEqual(cli.get_config_dir(), Path.home() / ".config" / "xpu_shell")
        self.assertEqual(cli.get_last_pod_id_file(), Path.home() / ".config" / "xpu_shell" / ".last_pod_id")

  @patch("subprocess.run")
  @patch("xpu_shell.cli.save_last_pod_id")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("runpod.get_pod")
  @patch("runpod.create_pod")
  def test_create_saves_last_pod_id(self, mock_create, mock_get_pod, mock_wait_ssh, mock_save, mock_sub_run):
    mock_create.return_value = {"id": "pod-new-456"}
    mock_get_pod.return_value = {
        "id": "pod-new-456",
        "status": "RUNNING",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "create",
        "--image-name", "runpod/pytorch",
        "--no-wait-for-setup"
    ]
    with patch("xpu_shell.cli.get_ssh_key", return_value="ssh-rsa fake"):
      with patch.object(sys, "argv", test_args):
        cli.main()

    mock_save.assert_called_once_with("pod-new-456")

  def test_save_and_get_last_job_id(self):
    with tempfile.TemporaryDirectory() as td:
      fake_file = Path(td) / ".last_job_id"
      with patch.dict(os.environ, {"XPU_SHELL_LAST_JOB_ID_FILE": str(fake_file)}):
        self.assertIsNone(cli.get_last_job_id())
        cli.save_last_job_id("job-abc-123")
        self.assertEqual(cli.get_last_job_id(), "job-abc-123")
        cli.clear_last_job_id_if_match("job-other")
        self.assertEqual(cli.get_last_job_id(), "job-abc-123")
        cli.clear_last_job_id_if_match("job-abc-123")
        self.assertIsNone(cli.get_last_job_id())

  def test_job_id_file_env_overrides(self):
    with tempfile.TemporaryDirectory() as td:
      dir_path = Path(td) / "custom_config"
      file_path = Path(td) / "custom_file"
      with patch.dict(os.environ, {"XPU_SHELL_CONFIG_DIR": str(dir_path)}, clear=True):
        self.assertEqual(cli.get_last_job_id_file(), dir_path / ".last_job_id")

      with patch.dict(os.environ, {"XPU_SHELL_LAST_JOB_ID_FILE": str(file_path)}, clear=True):
        self.assertEqual(cli.get_last_job_id_file(), file_path)

      with patch.dict(os.environ, {}, clear=True):
        self.assertEqual(cli.get_last_job_id_file(), Path.home() / ".config" / "xpu_shell" / ".last_job_id")



  @patch("xpu_shell.cli.save_last_job_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_command")
  @patch("runpod.get_pod")
  def test_run_command_with_pod_flag_and_args(self, mock_get_pod, mock_exec_cmd, mock_find_priv, mock_save_job):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec_cmd.return_value = {"job_id": "job-run-1", "pid": "999", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "run",
        "-p", "pod-123",
        "-d",
        "python3", "-c", "print('hello')"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_get_pod.assert_called_once_with("pod-123")
    mock_exec_cmd.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        command_args=["python3", "-c", "print('hello')"],
        detach=True,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env=None,
        use_shell=False,
        verbose=False
    )
    mock_save_job.assert_called_once_with("job-run-1")

  @patch("xpu_shell.cli.save_last_job_id")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_command")
  @patch("runpod.get_pod")
  def test_run_command_with_last_pod_id(self, mock_get_pod, mock_exec_cmd, mock_find_priv, mock_last_pod, mock_save_job):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec_cmd.return_value = {"job_id": "job-run-2", "pid": "1000", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "run",
        "ls", "-la"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_get_pod.assert_called_once_with("pod-last")
    mock_exec_cmd.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        command_args=["ls", "-la"],
        detach=False,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env=None,
        use_shell=False,
        verbose=False
    )
    mock_save_job.assert_called_once_with("job-run-2")

  @patch("xpu_shell.cli.save_last_job_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_saves_last_job_id(self, mock_get_pod, mock_exec, mock_find_priv, mock_save_job):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-exec",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-exec-1", "pid": "101", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "-p", "pod-exec",
        "train.py"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_save_job.assert_called_once_with("job-exec-1")

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.view_remote_logs")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_logs_with_job_flag(self, mock_get_pod, mock_last_pod, mock_logs, mock_find_priv):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = ["cli.py", "--api-key", "fake-api-key", "logs", "-j", "job-flag-1"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")
    mock_logs.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        job_id="job-flag-1",
        tail_lines=None,
        follow=False,
        private_key_path=None,
        ssh_config_path=None,
        verbose=False
    )

  @patch("xpu_shell.cli.get_last_job_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.view_remote_logs")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_logs_with_last_job_id(self, mock_get_pod, mock_last_pod, mock_logs, mock_find_priv, mock_last_job):
    mock_last_pod.return_value = "pod-last"
    mock_last_job.return_value = "job-last-saved"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = ["cli.py", "--api-key", "fake-api-key", "logs"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")
    mock_logs.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        job_id="job-last-saved",
        tail_lines=None,
        follow=False,
        private_key_path=None,
        ssh_config_path=None,
        verbose=False
    )

  @patch("xpu_shell.cli.clear_last_job_id_if_match")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.kill_remote_job")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_kill_with_job_flag(self, mock_get_pod, mock_last_pod, mock_kill, mock_find_priv, mock_clear_job):
    mock_last_pod.return_value = "pod-last"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = ["cli.py", "--api-key", "fake-api-key", "kill", "-j", "job-kill-1"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")
    mock_kill.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        target_id="job-kill-1",
        signal_name="SIGTERM",
        timeout=30.0,
        private_key_path=None,
        ssh_config_path=None
    )
    mock_clear_job.assert_called_once_with("job-kill-1")

  @patch("xpu_shell.cli.get_last_job_id")
  @patch("xpu_shell.cli.clear_last_job_id_if_match")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.kill_remote_job")
  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("runpod.get_pod")
  def test_kill_with_last_job_id(self, mock_get_pod, mock_last_pod, mock_kill, mock_find_priv, mock_clear_job, mock_last_job):
    mock_last_pod.return_value = "pod-last"
    mock_last_job.return_value = "job-last-killed"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-last",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    test_args = ["cli.py", "--api-key", "fake-api-key", "kill"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_get_pod.assert_called_once_with("pod-last")
    mock_kill.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        target_id="job-last-killed",
        signal_name="SIGTERM",
        timeout=30.0,
        private_key_path=None,
        ssh_config_path=None
    )
    mock_clear_job.assert_called_once_with("job-last-killed")

  @patch("runpod.stop_pod")
  def test_stop_with_pod_short_flag(self, mock_stop):
    test_args = ["cli.py", "--api-key", "fake-api-key", "stop", "-p", "pod-short"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_stop.assert_called_once_with("pod-short")

  @patch("runpod.terminate_pod")
  def test_terminate_with_pod_short_flag(self, mock_term):
    test_args = ["cli.py", "--api-key", "fake-api-key", "terminate", "-p", "pod-short"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    mock_term.assert_called_once_with("pod-short")

  @patch("xpu_shell.cli.save_last_job_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_command")
  @patch("runpod.get_pod")
  def test_run_command_with_shell_flag(self, mock_get_pod, mock_exec_cmd, mock_find_priv, mock_save_job):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec_cmd.return_value = {"job_id": "job-shell-1", "pid": "100", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "run",
        "-p", "pod-123",
        "-s",
        "mv", "/workspace/temp/co*", "/workspace/"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_exec_cmd.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        command_args=["mv", "/workspace/temp/co*", "/workspace/"],
        detach=False,
        private_key_path=None,
        wait_for_setup_flag=True,
        ssh_timeout=180,
        ssh_config_path=None,
        extra_env=None,
        use_shell=True,
        verbose=False
    )

  @patch("subprocess.run")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("runpod.get_pod")
  def test_cp_local_to_remote(self, mock_get_pod, mock_find_priv, mock_subproc):
    mock_find_priv.return_value = Path("/my/key")
    mock_get_pod.return_value = {
        "id": "pod-cp-1",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 54321, "address": "98.76.54.32"}]
        }
    }
    mock_subproc.return_value = MagicMock(returncode=0)

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "cp",
        "-p", "pod-cp-1",
        "local_file.txt",
        ":/workspace/remote_file.txt"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_get_pod.assert_called_once_with("pod-cp-1")
    mock_subproc.assert_called_once()
    called_cmd = mock_subproc.call_args[0][0]
    self.assertEqual(called_cmd[0], "scp")
    self.assertIn("-P", called_cmd)
    self.assertIn("54321", called_cmd)
    self.assertEqual(called_cmd[-2], "local_file.txt")
    self.assertEqual(called_cmd[-1], "root@98.76.54.32:/workspace/remote_file.txt")

  @patch("subprocess.run")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("runpod.get_pod")
  def test_cp_remote_to_local_with_prefix_pod(self, mock_get_pod, mock_find_priv, mock_subproc):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "custom-pod",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 2222, "address": "1.2.3.4"}]
        }
    }
    mock_subproc.return_value = MagicMock(returncode=0)

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "cp",
        "-r",
        "-P",
        "custom-pod:/workspace/data",
        "./local_data"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_get_pod.assert_called_once_with("custom-pod")
    called_cmd = mock_subproc.call_args[0][0]
    self.assertIn("-r", called_cmd)
    self.assertIn("-p", called_cmd)
    self.assertEqual(called_cmd[-2], "root@1.2.3.4:/workspace/data")
    self.assertEqual(called_cmd[-1], "./local_data")

  @patch("xpu_shell.cli.get_last_pod_id")
  @patch("subprocess.run")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("runpod.get_pod")
  def test_cp_with_last_pod(self, mock_get_pod, mock_find_priv, mock_subproc, mock_last_pod):
    mock_last_pod.return_value = "last-pod-99"
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "last-pod-99",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 2222, "address": "1.2.3.4"}]
        }
    }
    mock_subproc.return_value = MagicMock(returncode=0)

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "cp",
        ":/workspace/temp/co*",
        "./dest/"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_get_pod.assert_called_once_with("last-pod-99")
    called_cmd = mock_subproc.call_args[0][0]
    self.assertEqual(called_cmd[-2], "root@1.2.3.4:/workspace/temp/co*")
    self.assertEqual(called_cmd[-1], "./dest/")

  def test_cp_validation_errors(self):
    # No remote path
    test_args = ["cli.py", "--api-key", "fake-api-key", "cp", "local1", "local2"]
    with patch.object(sys, "argv", test_args):
      with self.assertRaises(RuntimeError):
        cli.main()

    # Both remote paths
    test_args2 = ["cli.py", "--api-key", "fake-api-key", "cp", ":/remote1", ":/remote2"]
    with patch.object(sys, "argv", test_args2):
      with self.assertRaises(RuntimeError):
        cli.main()

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_script")
  @patch("runpod.get_pod")
  def test_exec_with_verbose_flag(self, mock_get_pod, mock_exec, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "exec",
        "-v",
        "pod-123",
        "script.sh"
    ]

    with patch("sys.stderr") as mock_stderr:
      with patch.object(sys, "argv", test_args):
        cli.main()

      mock_exec.assert_called_once_with(
          host="12.34.56.78",
          port=12345,
          script_path="script.sh",
          script_args="",
          detach=False,
          private_key_path=None,
          wait_for_setup_flag=True,
          ssh_timeout=180,
          ssh_config_path=None,
          extra_env=None,
          verbose=True
      )
      self.assertTrue(any("Fetching connection details" in str(c) for c in mock_stderr.method_calls))

  @patch("xpu_shell.cli.save_last_job_id")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.execute_remote_command")
  @patch("runpod.get_pod")
  def test_run_with_verbose_flag(self, mock_get_pod, mock_exec_cmd, mock_find_priv, mock_save_job):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }
    mock_exec_cmd.return_value = {"job_id": "job-1", "pid": "123", "exit_code": 0}

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "run",
        "-v",
        "-p", "pod-123",
        "echo", "hi"
    ]

    with patch("sys.stderr") as mock_stderr:
      with patch.object(sys, "argv", test_args):
        cli.main()

      mock_exec_cmd.assert_called_once_with(
          host="12.34.56.78",
          port=12345,
          command_args=["echo", "hi"],
          detach=False,
          private_key_path=None,
          wait_for_setup_flag=True,
          ssh_timeout=180,
          ssh_config_path=None,
          extra_env=None,
          use_shell=False,
          verbose=True
      )
      self.assertTrue(any("Fetching connection details" in str(c) for c in mock_stderr.method_calls))

  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.view_remote_logs")
  @patch("runpod.get_pod")
  def test_logs_with_verbose_flag(self, mock_get_pod, mock_view_logs, mock_find_priv):
    mock_find_priv.return_value = None
    mock_get_pod.return_value = {
        "id": "pod-123",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 12345, "address": "12.34.56.78"}]
        }
    }

    test_args = [
        "cli.py",
        "--api-key", "fake-api-key",
        "logs",
        "-v",
        "pod-123",
        "job-1"
    ]

    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_view_logs.assert_called_once_with(
        host="12.34.56.78",
        port=12345,
        job_id="job-1",
        tail_lines=None,
        follow=False,
        private_key_path=None,
        ssh_config_path=None,
        verbose=True
    )

  def test_format_uptime(self):
    self.assertEqual(cli.format_uptime(None), "N/A")
    self.assertEqual(cli.format_uptime(-1), "N/A")
    self.assertEqual(cli.format_uptime(45), "45s")
    self.assertEqual(cli.format_uptime(125), "2m 5s")
    self.assertEqual(cli.format_uptime(3665), "1h 1m 5s")
    self.assertEqual(cli.format_uptime(90061), "1d 1h 1m 1s")

  @patch("xpu_shell.cli.run_graphql")
  @patch("runpod.get_gpus")
  def test_info_command_formatted_output(self, mock_get_gpus, mock_run_graphql):
    mock_get_gpus.return_value = [
        {"id": "NVIDIA GeForce RTX 4090", "displayName": "RTX 4090", "memoryInGb": 24}
    ]
    mock_run_graphql.return_value = {
        "data": {
            "pod": {
                "id": "pod-123",
                "name": "my-worker",
                "desiredStatus": "RUNNING",
                "createdAt": "2026-09-10T10:00:00.000Z",
                "costPerHr": 0.74,
                "podType": "COMMUNITY",
                "imageName": "runpod/pytorch:latest",
                "vcpuCount": 16,
                "memoryInGb": 64,
                "containerDiskInGb": 50,
                "volumeInGb": 100,
                "volumeMountPath": "/workspace",
                "gpuCount": 1,
                "runtime": {
                    "uptimeInSeconds": 3665,
                    "ports": [
                        {"privatePort": 22, "publicPort": 2222, "ip": "1.2.3.4"}
                    ],
                    "container": {"cpuPercent": 15, "memoryPercent": 25},
                    "gpus": [{"gpuUtilPercent": 98, "memoryUtilPercent": 80}]
                },
                "machine": {
                    "gpuDisplayName": "RTX 4090",
                    "gpuTypeId": "NVIDIA GeForce RTX 4090",
                    "location": "US",
                    "dataCenterId": "US-NC-1"
                }
            }
        }
    }

    test_args = ["cli.py", "--api-key", "fake-key", "info", "-p", "pod-123"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    output = "\n".join(call[0][0] for call in mock_print.call_args_list if call[0])
    self.assertIn("my-worker (pod-123)", output)
    self.assertIn("RUNNING (COMMUNITY)", output)
    self.assertIn("1h 1m 5s", output)
    self.assertIn("1x RTX 4090 (24 GB VRAM)", output)
    self.assertIn("16 vCPUs", output)
    self.assertIn("64 GB", output)
    self.assertIn("50 GB", output)
    self.assertIn("100 GB (mounted at /workspace)", output)
    self.assertIn("CPU Usage:      15%", output)
    self.assertIn("RAM Usage:      25%", output)
    self.assertIn("GPU Compute:    98%", output)
    self.assertIn("GPU VRAM:       80%", output)
    self.assertIn("1.2.3.4:2222", output)
    self.assertIn("ssh -p 2222 root@1.2.3.4", output)

  @patch("xpu_shell.cli.run_graphql")
  def test_info_command_json_output(self, mock_run_graphql):
    mock_run_graphql.return_value = {
        "data": {
            "pod": {
                "id": "pod-123",
                "name": "my-worker",
                "desiredStatus": "RUNNING"
            }
        }
    }

    test_args = ["cli.py", "--api-key", "fake-key", "info", "--json", "pod-123"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_print.assert_called_once()
    printed_json = mock_print.call_args[0][0]
    self.assertIn('"id": "pod-123"', printed_json)
    self.assertIn('"name": "my-worker"', printed_json)

  @patch("xpu_shell.cli.run_graphql", side_effect=Exception("GraphQL error"))
  @patch("runpod.get_pod")
  def test_info_command_fallback_to_get_pod(self, mock_get_pod, mock_run_graphql):
    mock_get_pod.return_value = {
        "id": "pod-fallback",
        "name": "fallback-worker",
        "desiredStatus": "RUNNING",
        "vcpuCount": 4,
        "memoryInGb": 16,
        "gpuCount": 0
    }

    test_args = ["cli.py", "--api-key", "fake-key", "info", "-p", "pod-fallback"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    output = "\n".join(call[0][0] for call in mock_print.call_args_list if call[0])
    self.assertIn("fallback-worker (pod-fallback)", output)
    self.assertIn("CPU only (no dedicated GPU)", output)
    self.assertIn("4 vCPUs", output)
    self.assertIn("16 GB", output)

  @patch("xpu_shell.cli.run_graphql", side_effect=Exception("GraphQL error"))
  @patch("runpod.get_pod", return_value=None)
  def test_info_command_pod_not_found(self, mock_get_pod, mock_run_graphql):
    test_args = ["cli.py", "--api-key", "fake-key", "info", "-p", "nonexistent-pod"]
    with patch.object(sys, "argv", test_args):
      with self.assertRaises(FileNotFoundError):
        cli.main()


if __name__ == "__main__":
  unittest.main()

