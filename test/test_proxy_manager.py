import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import xpu_shell.proxy_manager as pm


class TestProxyManager(unittest.TestCase):

  def setUp(self):
    self.temp_dir = tempfile.mkdtemp()
    self.patcher = patch.dict(os.environ, {"XPU_SHELL_CONFIG_DIR": self.temp_dir})
    self.patcher.start()

  def tearDown(self):
    self.patcher.stop()
    shutil.rmtree(self.temp_dir, ignore_errors=True)

  def test_get_proxy_dir(self):
    base_dir = pm.get_proxy_base_dir()
    self.assertEqual(base_dir, Path(self.temp_dir) / "proxies")
    pod_dir = pm.get_proxy_dir("pod-123")
    self.assertEqual(pod_dir, Path(self.temp_dir) / "proxies" / "pod-123")

  def test_is_pid_alive(self):
    self.assertTrue(pm.is_pid_alive(os.getpid()))
    self.assertFalse(pm.is_pid_alive(99999999))
    self.assertFalse(pm.is_pid_alive(0))
    self.assertFalse(pm.is_pid_alive(-1))

  def test_get_active_proxy_info_live(self):
    pod_dir = pm.get_proxy_dir("pod-live")
    pod_dir.mkdir(parents=True)
    meta = {
      "pid": os.getpid(),
      "pod_id": "pod-live",
      "remote_port": 8080,
      "local_port": 39123,
      "routes": ["/hf"],
      "env": {"HF_ENDPOINT": "http://127.0.0.1:8080/hf"}
    }
    (pod_dir / "meta.json").write_text(json.dumps(meta))

    info = pm.get_active_proxy_info("pod-live")
    self.assertIsNotNone(info)
    self.assertEqual(info["pod_id"], "pod-live")
    self.assertEqual(info["remote_port"], 8080)

  def test_get_active_proxy_info_dead_cleans_up(self):
    pod_dir = pm.get_proxy_dir("pod-dead")
    pod_dir.mkdir(parents=True)
    meta = {
      "pid": 99999999,
      "pod_id": "pod-dead"
    }
    (pod_dir / "meta.json").write_text(json.dumps(meta))

    info = pm.get_active_proxy_info("pod-dead")
    self.assertIsNone(info)
    self.assertFalse(pod_dir.exists())

  def test_list_active_proxies(self):
    pod_dir1 = pm.get_proxy_dir("pod-1")
    pod_dir1.mkdir(parents=True)
    meta1 = {"pid": os.getpid(), "pod_id": "pod-1", "remote_port": 8080, "routes": []}
    (pod_dir1 / "meta.json").write_text(json.dumps(meta1))

    pod_dir2 = pm.get_proxy_dir("pod-2")
    pod_dir2.mkdir(parents=True)
    meta2 = {"pid": 99999999, "pod_id": "pod-2"}
    (pod_dir2 / "meta.json").write_text(json.dumps(meta2))

    active = pm.list_active_proxies()
    self.assertEqual(len(active), 1)
    self.assertEqual(active[0]["pod_id"], "pod-1")

  @patch("subprocess.run")
  def test_deploy_remote_profile(self, mock_run):
    pm.deploy_remote_profile(
      host="1.2.3.4",
      port=22,
      env_vars={"HF_ENDPOINT": "http://127.0.0.1:8080/hf"}
    )
    self.assertTrue(mock_run.called)
    call_args = mock_run.call_args[0][0]
    self.assertIn("ssh", call_args[0])
    remote_cmd = call_args[-1]
    self.assertIn("/etc/profile.d/xpu_proxy.sh", remote_cmd)
    self.assertIn("HF_ENDPOINT", remote_cmd)

  @patch("subprocess.run")
  def test_remove_remote_profile(self, mock_run):
    pm.remove_remote_profile(host="1.2.3.4", port=22)
    self.assertTrue(mock_run.called)
    call_args = mock_run.call_args[0][0]
    self.assertIn("rm -f /etc/profile.d/xpu_proxy.sh", call_args[-1])

  @patch("xpu_shell.proxy_manager.deploy_remote_profile")
  @patch("subprocess.Popen")
  def test_start_proxy_for_pod(self, mock_popen, mock_deploy):
    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.pid = os.getpid()
    mock_popen.return_value = mock_proc

    # Simulate worker writing meta.json
    def side_effect(*args, **kwargs):
      pod_dir = pm.get_proxy_dir("pod-new")
      meta = {
        "pid": os.getpid(),
        "pod_id": "pod-new",
        "remote_port": 8080,
        "local_port": 39123,
        "routes": ["/hf"],
        "env": {"HF_ENDPOINT": "http://127.0.0.1:8080/hf"}
      }
      (pod_dir / "meta.json").write_text(json.dumps(meta))
      return mock_proc

    mock_popen.side_effect = side_effect

    info = pm.start_proxy_for_pod(
      pod_id="pod-new",
      host="1.2.3.4",
      port=22,
      remote_port=8080
    )
    self.assertEqual(info["pod_id"], "pod-new")
    self.assertTrue(mock_deploy.called)

  @patch("xpu_shell.proxy_manager.remove_remote_profile")
  @patch("os.kill")
  def test_stop_proxy_for_pod(self, mock_kill, mock_remove):
    pod_dir = pm.get_proxy_dir("pod-stop")
    pod_dir.mkdir(parents=True)
    meta = {
      "pid": 54321,
      "pod_id": "pod-stop",
      "host": "1.2.3.4",
      "port": 22
    }
    (pod_dir / "meta.json").write_text(json.dumps(meta))

    with patch("xpu_shell.proxy_manager.is_pid_alive") as mock_alive:
      # True for info lookup and signal check, then False (terminated)
      mock_alive.side_effect = [True, True, False, False]
      stopped = pm.stop_proxy_for_pod("pod-stop")
      self.assertTrue(stopped)
      self.assertTrue(mock_kill.called)
      self.assertTrue(mock_remove.called)
      self.assertFalse(pod_dir.exists())


if __name__ == "__main__":
  unittest.main()
