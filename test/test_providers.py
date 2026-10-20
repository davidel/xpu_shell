import os
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import MagicMock, patch

import xpu_shell.cli as cli


from xpu_shell.providers import (
    get_provider,
    resolve_provider,
    parse_instance_identifier,
    save_last_provider,
    get_last_provider
)
from xpu_shell.providers.base import Instance
from xpu_shell.providers.runpod_provider import RunPodProvider
from xpu_shell.providers.vast_provider import VastProvider


class TestProviders(unittest.TestCase):

  def test_parse_instance_identifier(self):
    prov, pid = parse_instance_identifier("runpod:abc-123")
    self.assertEqual(prov, "runpod")
    self.assertEqual(pid, "abc-123")

    prov, pid = parse_instance_identifier("vast:98765")
    self.assertEqual(prov, "vast")
    self.assertEqual(pid, "98765")

    prov, pid = parse_instance_identifier("plain-id")
    self.assertIsNone(prov)
    self.assertEqual(pid, "plain-id")

  def test_get_provider(self):
    rp = get_provider("runpod")
    self.assertIsInstance(rp, RunPodProvider)
    self.assertEqual(rp.name, "runpod")

    vast = get_provider("vast", api_key="test-key")
    self.assertIsInstance(vast, VastProvider)
    self.assertEqual(vast.name, "vast")

    with self.assertRaises(ValueError):
      get_provider("invalid-provider")

  def test_resolve_provider_precedence(self):
    # 1. Explicit CLI argument
    args = MagicMock(provider="vast", api_key="vast-key")
    p = resolve_provider(args=args)
    self.assertIsInstance(p, VastProvider)

    # 2. Prefixed ID
    p = resolve_provider(instance_id="vast:12345")
    self.assertIsInstance(p, VastProvider)

    p = resolve_provider(instance_id="runpod:xyz-999")
    self.assertIsInstance(p, RunPodProvider)

    # 3. Numeric ID heuristic
    p = resolve_provider(instance_id="849203")
    self.assertIsInstance(p, VastProvider)

    # 4. State file
    with tempfile.TemporaryDirectory() as td:
      with patch.dict(os.environ, {"XPU_SHELL_CONFIG_DIR": td}):
        save_last_provider("vast")
        self.assertEqual(get_last_provider(), "vast")
        p = resolve_provider(instance_id="alphanumeric-id")
        self.assertIsInstance(p, VastProvider)

    # 5. Default fallback
    with patch.dict(os.environ, {}, clear=True):
      with patch("xpu_shell.providers.get_last_provider", return_value=None):
        p = resolve_provider(instance_id="alpha-node")
        self.assertIsInstance(p, RunPodProvider)


class TestRunPodProvider(unittest.TestCase):

  @patch("runpod.get_pods")
  def test_list_instances(self, mock_get_pods):
    mock_get_pods.return_value = [
        {
            "id": "pod-1",
            "name": "worker-1",
            "desiredStatus": "RUNNING",
            "costPerHr": 0.44,
            "runtime": {
                "ports": [{"privatePort": 22, "publicPort": 10022, "ip": "1.2.3.4"}]
            }
        }
    ]
    rp = RunPodProvider()
    instances = rp.list_instances()
    self.assertEqual(len(instances), 1)
    inst = instances[0]
    self.assertEqual(inst.id, "pod-1")
    self.assertEqual(inst.name, "worker-1")
    self.assertEqual(inst.status, "RUNNING")
    self.assertEqual(inst.ssh_host, "1.2.3.4")
    self.assertEqual(inst.ssh_port, 10022)
    self.assertEqual(inst.cost_per_hr, 0.44)

  @patch("runpod.get_pod")
  def test_get_ssh_endpoint(self, mock_get_pod):
    mock_get_pod.return_value = {
        "id": "pod-2",
        "runtime": {
            "ports": [{"privatePort": 22, "isExternal": 2222, "address": "5.6.7.8"}]
        }
    }
    rp = RunPodProvider()
    host, port = rp.get_ssh_endpoint("pod-2")
    self.assertEqual(host, "5.6.7.8")
    self.assertEqual(port, 2222)


class TestVastProvider(unittest.TestCase):

  @patch("requests.Session.request")
  def test_list_instances(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"instances": []}'
    mock_resp.json.return_value = {
        "instances": [
            {
                "id": 12345,
                "label": "vast-gpu-node",
                "actual_status": "running",
                "ssh_host": "ssh3.vast.ai",
                "ssh_port": 34567,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "dph_total": 0.35,
                "disk_space": 100.0,
                "image": "pytorch/pytorch:latest"
            }
        ]
    }
    mock_req.return_value = mock_resp

    vast = VastProvider(api_key="fake-vast-key")
    instances = vast.list_instances()
    self.assertEqual(len(instances), 1)
    called_url = mock_req.call_args[0][1]
    self.assertIn("/api/v1/instances/", called_url)
    inst = instances[0]
    self.assertEqual(inst.id, "12345")
    self.assertEqual(inst.name, "vast-gpu-node")
    self.assertEqual(inst.status, "RUNNING")
    self.assertEqual(inst.ssh_host, "ssh3.vast.ai")
    self.assertEqual(inst.ssh_port, 34567)
    self.assertEqual(inst.gpu_name, "RTX 4090")
    self.assertEqual(inst.cost_per_hr, 0.35)

  @patch("requests.Session.request")
  def test_search_offers(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"offers": []}'
    mock_resp.json.return_value = {
        "offers": [
            {
                "id": 9991,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "rented": False,
                "reliability2": 0.98,
                "dph_total": 0.40,
                "disk_space": 100.0
            },
            {
                "id": 9992,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "rented": False,
                "reliability2": 0.99,
                "dph_total": 0.30,
                "disk_space": 100.0
            },
            {
                "id": 9993,
                "gpu_name": "RTX 3090",
                "num_gpus": 1,
                "rented": False,
                "reliability2": 0.99,
                "dph_total": 0.20,
                "disk_space": 50.0
            }
        ]
    }
    mock_req.return_value = mock_resp

    vast = VastProvider(api_key="fake-vast-key")
    offers = vast.search_offers(gpu_name="4090")
    self.assertEqual(len(offers), 2)
    # Sorted by price ascending:
    self.assertEqual(offers[0]["id"], 9992)
    self.assertEqual(offers[0]["dph_total"], 0.30)
    self.assertEqual(offers[1]["id"], 9991)

  @patch("requests.Session.request")
  def test_stop_and_terminate(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{}'
    mock_resp.json.return_value = {"success": True}
    mock_req.return_value = mock_resp

    vast = VastProvider(api_key="fake-vast-key")
    vast.stop_instance("12345")
    vast.terminate_instance("12345")
    self.assertEqual(mock_req.call_count, 2)


class TestVastCLIIntegration(unittest.TestCase):

  @patch("xpu_shell.cli.resolve_provider")
  @patch("sys.stdout")
  def test_vast_cli_list(self, mock_stdout, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_prov.list_instances.return_value = [
        Instance(
            id="12345",
            name="vast-node",
            provider="vast",
            status="RUNNING",
            ssh_host="ssh.vast.ai",
            ssh_port=2222,
            gpu_name="RTX 4090",
            gpu_count=1
        )
    ]
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "--provider", "vast", "--api-key", "fake-key", "list"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("12345", output)
    self.assertIn("vast-node", output)
    self.assertIn("ssh.vast.ai:2222", output)

  @patch("xpu_shell.cli.resolve_provider")
  def test_vast_cli_stop(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "--provider", "vast", "--api-key", "fake-key", "stop", "--pod", "12345"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_prov.stop_instance.assert_called_once_with("12345")

  @patch("xpu_shell.cli.resolve_provider")
  def test_vast_cli_terminate(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "--provider", "vast", "--api-key", "fake-key", "terminate", "--pod", "12345"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_prov.terminate_instance.assert_called_once_with("12345")

  @patch("xpu_shell.cli.execute_remote_script")
  @patch("xpu_shell.cli.resolve_instance_ssh_endpoint")
  @patch("xpu_shell.cli.find_ssh_private_key")
  def test_vast_cli_exec(self, mock_key, mock_endpoint, mock_exec):
    mock_endpoint.return_value = ("ssh.vast.ai", 2222)
    mock_key.return_value = None
    mock_exec.return_value = {"job_id": "job-1", "exit_code": 0}

    test_args = [
        "cli.py",
        "--provider", "vast",
        "--api-key", "fake-key",
        "exec",
        "--pod", "12345",
        "train.py",
        "--ssh-config", "/dev/null"
    ]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_exec.assert_called_once()
    self.assertEqual(mock_exec.call_args[1]["host"], "ssh.vast.ai")
    self.assertEqual(mock_exec.call_args[1]["port"], 2222)

  @patch("xpu_shell.cli.resolve_provider")
  @patch("sys.stdout")
  def test_vast_cli_gpus(self, mock_stdout, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_prov.list_gpus.return_value = [
        {
            "id": "RTX 4090",
            "displayName": "RTX 4090",
            "memoryInGb": 24,
            "minPrice": 0.29,
            "maxPrice": 0.45,
            "count": 12
        }
    ]
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "--provider", "vast", "--api-key", "fake-key", "gpus"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("RTX 4090", output)
    self.assertIn("$0.29/hr", output)
    self.assertIn("12 offers", output)

  @patch("xpu_shell.cli.resolve_provider")
  @patch("sys.stdout")
  def test_vast_cli_gpus_expand(self, mock_stdout, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_prov.list_gpus.return_value = [
        {
            "id": 9992,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "gpu_ram": 24576,
            "dph_total": 0.345,
            "reliability2": 0.992,
            "inet_down": 1200.0,
            "inet_up": 800.0,
            "disk_space": 150.0,
            "geolocation": "US"
        }
    ]
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "--provider", "vast", "--api-key", "fake-key", "gpus", "-l", "4090"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_prov.list_gpus.assert_called_with("4090", expand=True)
    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("OFFER ID", output)
    self.assertIn("9992", output)
    self.assertIn("RTX 4090", output)
    self.assertIn("$0.345/hr", output)
    self.assertIn("99.2%", output)
    self.assertIn("1200/800 Mbps", output)
    self.assertIn("150 GB", output)
    self.assertIn("US", output)

  @patch("requests.Session.request")
  def test_vast_provider_list_gpus_expand(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"offers": []}'
    mock_resp.json.return_value = {
        "offers": [
            {
                "id": 1111,
                "gpu_name": "RTX 4090",
                "num_gpus": 1,
                "rented": False,
                "reliability2": 0.98,
                "dph_total": 0.40,
                "disk_space": 100.0
            },
            {
                "id": 2222,
                "gpu_name": "RTX 3090",
                "num_gpus": 1,
                "rented": False,
                "reliability2": 0.95,
                "dph_total": 0.20,
                "disk_space": 50.0
            }
        ]
    }
    mock_req.return_value = mock_resp

    vast = VastProvider(api_key="fake-key")
    offers = vast.list_gpus("4090", expand=True)
    self.assertEqual(len(offers), 1)
    self.assertEqual(offers[0]["id"], 1111)

