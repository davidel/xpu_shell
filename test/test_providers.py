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
from xpu_shell.providers.spheron_provider import SpheronProvider


class TestProviders(unittest.TestCase):

  def test_parse_instance_identifier(self):
    prov, pid = parse_instance_identifier("runpod:abc-123")
    self.assertEqual(prov, "runpod")
    self.assertEqual(pid, "abc-123")

    prov, pid = parse_instance_identifier("vast:98765")
    self.assertEqual(prov, "vast")
    self.assertEqual(pid, "98765")

    prov, pid = parse_instance_identifier("spheron:dep-777")
    self.assertEqual(prov, "spheron")
    self.assertEqual(pid, "dep-777")

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

    spheron = get_provider("spheron", api_key="test-key")
    self.assertIsInstance(spheron, SpheronProvider)
    self.assertEqual(spheron.name, "spheron")

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

    p = resolve_provider(instance_id="spheron:dep-111")
    self.assertIsInstance(p, SpheronProvider)

    p = resolve_provider(instance_id="spheron-node-222")
    self.assertIsInstance(p, SpheronProvider)

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

  @patch("requests.Session.request")
  def test_create_instance_default_image(self, mock_req):
    mock_put_resp = MagicMock()
    mock_put_resp.status_code = 200
    mock_put_resp.content = b'{}'
    mock_put_resp.json.return_value = {"new_contract": 5555}

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{}'
    mock_get_resp.json.return_value = {
        "instances": [
            {
                "id": 5555,
                "actual_status": "running",
                "ssh_host": "ssh.vast.ai",
                "ssh_port": 2222
            }
        ]
    }
    mock_req.side_effect = [mock_put_resp, mock_get_resp]

    vast = VastProvider(api_key="fake-vast-key")
    self.assertEqual(vast.default_image, "pytorch/pytorch:latest")
    inst = vast.create_instance(offer_id=1234)
    self.assertEqual(inst.id, "5555")

    put_call = mock_req.call_args_list[0]
    self.assertEqual(put_call[0][0], "PUT")
    self.assertIn("/asks/1234/", put_call[0][1])
    body = put_call[1]["json"]
    self.assertEqual(body["image"], "pytorch/pytorch:latest")


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

  def test_runpod_rejects_offer_id(self):
    test_args = ["cli.py", "create", "--provider", "runpod", "--offer-id", "12345"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  def test_vast_rejects_template_id(self):
    test_args = ["cli.py", "create", "--provider", "vast", "--template-id", "runpod-pytorch"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  def test_vast_rejects_cloud_type(self):
    test_args = ["cli.py", "create", "--provider", "vast", "--cloud-type", "COMMUNITY"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  def test_vast_rejects_templates_subcommand(self):
    test_args = ["cli.py", "--provider", "vast", "templates"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  @patch("xpu_shell.providers.vast_provider.VastProvider.create_instance")
  @patch("xpu_shell.providers.vast_provider.VastProvider.get_instance")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("time.sleep")
  def test_create_with_offer_id_auto_detects_vast(self, mock_sleep, mock_ssh_key, mock_find_priv, mock_wait_ssh, mock_get_inst, mock_create_inst):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    fake_inst = Instance(
        id="12345",
        name="test-worker",
        provider="vast",
        status="RUNNING",
        ssh_host="ssh.vast.ai",
        ssh_port=2222
    )
    mock_create_inst.return_value = fake_inst
    mock_get_inst.return_value = fake_inst

    test_args = ["cli.py", "create", "--offer-id", "9999", "--image-name", "pytorch/pytorch:latest"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_create_inst.assert_called_once()
    self.assertEqual(mock_create_inst.call_args[1]["offer_id"], 9999)

  @patch("xpu_shell.providers.vast_provider.VastProvider.create_instance")
  @patch("xpu_shell.providers.vast_provider.VastProvider.get_instance")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("time.sleep")
  def test_create_without_image_name_passes_none_to_vast_provider(self, mock_sleep, mock_ssh_key, mock_find_priv, mock_wait_ssh, mock_get_inst, mock_create_inst):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    fake_inst = Instance(
        id="12345",
        name="test-worker",
        provider="vast",
        status="RUNNING",
        ssh_host="ssh.vast.ai",
        ssh_port=2222
    )
    mock_create_inst.return_value = fake_inst
    mock_get_inst.return_value = fake_inst

    test_args = ["cli.py", "create", "--offer-id", "9999"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_create_inst.assert_called_once()
    self.assertIsNone(mock_create_inst.call_args[1]["image_name"])


class TestSpheronProvider(unittest.TestCase):

  @patch("requests.Session.request")
  def test_list_instances(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"deployments": []}'
    mock_resp.json.return_value = {
        "deployments": [
            {
                "id": "spheron-dep-1",
                "name": "worker-spheron-1",
                "status": "Running",
                "pricePerHour": 0.55,
                "gpuType": "RTX 4090",
                "gpuCount": 1,
                "vcpu": 8,
                "memoryInGb": 32,
                "disk": 100,
                "operatingSystem": "ubuntu-22.04",
                "connectionDetails": {
                    "ip": "54.210.10.20",
                    "port": 22
                }
            }
        ]
    }
    mock_req.return_value = mock_resp

    sp = SpheronProvider(api_key="fake-key")
    instances = sp.list_instances()
    self.assertEqual(len(instances), 1)
    inst = instances[0]
    self.assertEqual(inst.id, "spheron-dep-1")
    self.assertEqual(inst.name, "worker-spheron-1")
    self.assertEqual(inst.provider, "spheron")
    self.assertEqual(inst.status, "RUNNING")
    self.assertEqual(inst.ssh_host, "54.210.10.20")
    self.assertEqual(inst.ssh_port, 22)
    self.assertEqual(inst.gpu_name, "RTX 4090")
    self.assertEqual(inst.gpu_count, 1)
    self.assertEqual(inst.cost_per_hr, 0.55)
    self.assertEqual(inst.vcpu_count, 8)
    self.assertEqual(inst.memory_gb, 32.0)
    self.assertEqual(inst.disk_gb, 100.0)

  @patch("requests.Session.request")
  def test_get_instance(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{}'
    mock_resp.json.return_value = {
        "id": "dep-456",
        "name": "my-spheron-node",
        "status": "deploying",
        "gpuType": "A100",
        "ip": "192.168.1.50"
    }
    mock_req.return_value = mock_resp

    sp = SpheronProvider(api_key="fake-key")
    inst = sp.get_instance("dep-456")
    self.assertEqual(inst.id, "dep-456")
    self.assertEqual(inst.status, "INITIALIZING")
    self.assertEqual(inst.ssh_host, "192.168.1.50")
    self.assertEqual(inst.ssh_port, 22)

  @patch("requests.Session.request")
  def test_create_instance_with_offer_id(self, mock_req):
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.content = b'{}'
    mock_post_resp.json.return_value = {"id": "new-dep-99"}

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{}'
    mock_get_resp.json.return_value = {
        "id": "new-dep-99",
        "name": "worker-new",
        "status": "running",
        "ip": "1.2.3.4"
    }

    mock_req.side_effect = [mock_post_resp, mock_get_resp]

    sp = SpheronProvider(api_key="fake-key")
    inst = sp.create_instance(
        name="worker-new",
        offer_id="spheron-offer-4090",
        ssh_key_id="ssh-key-xyz",
        gpu_count=1,
        disk_gb=80.0
    )
    self.assertEqual(inst.id, "new-dep-99")
    self.assertEqual(inst.status, "RUNNING")
    self.assertEqual(inst.ssh_host, "1.2.3.4")

    # Verify POST body
    post_call = mock_req.call_args_list[0]
    self.assertEqual(post_call[0][0], "POST")
    self.assertIn("/api/deployments", post_call[0][1])
    body = post_call[1]["json"]
    self.assertEqual(body["offerId"], "spheron-offer-4090")
    self.assertEqual(body["sshKeyId"], "ssh-key-xyz")
    self.assertEqual(body["disk"], 80.0)
    self.assertEqual(body["operatingSystem"], "ubuntu-22.04")
    self.assertEqual(sp.default_image, "ubuntu-22.04")

  @patch("requests.Session.request")
  def test_create_instance_auto_search(self, mock_req):
    mock_offers_resp = MagicMock()
    mock_offers_resp.status_code = 200
    mock_offers_resp.content = b'{}'
    mock_offers_resp.json.return_value = {
        "offers": [
            {
                "id": "offer-expensive",
                "gpuType": "RTX 4090",
                "pricePerHour": 0.80,
                "gpuCount": 1
            },
            {
                "id": "offer-cheap",
                "gpuType": "RTX 4090",
                "pricePerHour": 0.40,
                "gpuCount": 1
            }
        ]
    }

    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 200
    mock_post_resp.content = b'{}'
    mock_post_resp.json.return_value = {"id": "dep-auto"}

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{}'
    mock_get_resp.json.return_value = {
        "id": "dep-auto",
        "status": "deploying"
    }

    mock_req.side_effect = [mock_offers_resp, mock_post_resp, mock_get_resp]

    sp = SpheronProvider(api_key="fake-key")
    inst = sp.create_instance(gpu_name="RTX 4090")
    self.assertEqual(inst.id, "dep-auto")

    post_call = mock_req.call_args_list[1]
    body = post_call[1]["json"]
    self.assertEqual(body["offerId"], "offer-cheap")

  @patch("requests.Session.request")
  def test_stop_start_terminate(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{}'
    mock_resp.json.return_value = {}
    mock_req.return_value = mock_resp

    sp = SpheronProvider(api_key="fake-key")
    sp.stop_instance("dep-123")
    self.assertEqual(mock_req.call_args_list[0][0][0], "POST")
    self.assertIn("/api/deployments/dep-123/stop", mock_req.call_args_list[0][0][1])

    sp.start_instance("dep-123")
    self.assertEqual(mock_req.call_args_list[1][0][0], "POST")
    self.assertIn("/api/deployments/dep-123/start", mock_req.call_args_list[1][0][1])

    sp.terminate_instance("dep-123")
    self.assertEqual(mock_req.call_args_list[2][0][0], "DELETE")
    self.assertIn("/api/deployments/dep-123", mock_req.call_args_list[2][0][1])

  @patch("requests.Session.request")
  def test_list_gpus(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{}'
    mock_resp.json.return_value = {
        "offers": [
            {
                "id": "off-1",
                "gpuType": "RTX 4090",
                "pricePerHour": 0.45,
                "memoryInGb": 24,
                "gpuCount": 1
            },
            {
                "id": "off-2",
                "gpuType": "RTX 4090",
                "pricePerHour": 0.35,
                "memoryInGb": 24,
                "gpuCount": 1
            },
            {
                "id": "off-3",
                "gpuType": "A100",
                "pricePerHour": 1.50,
                "memoryInGb": 80,
                "gpuCount": 1
            }
        ]
    }
    mock_req.return_value = mock_resp

    sp = SpheronProvider(api_key="fake-key")
    gpus = sp.list_gpus()
    self.assertEqual(len(gpus), 2)
    rtx_agg = next(g for g in gpus if g["displayName"] == "RTX 4090")
    self.assertEqual(rtx_agg["count"], 2)
    self.assertEqual(rtx_agg["minPrice"], 0.35)
    self.assertEqual(rtx_agg["maxPrice"], 0.45)
    self.assertEqual(rtx_agg["memoryInGb"], 24)

  @patch("requests.Session.request")
  def test_list_gpus_expand(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{}'
    mock_resp.json.return_value = {
        "offers": [
            {
                "id": "spheron-4090-us",
                "gpuType": "RTX 4090",
                "pricePerHour": 0.42,
                "memoryInGb": 24,
                "gpuCount": 1,
                "disk": 100,
                "region": "us-east-1"
            },
            {
                "id": "spheron-a100-eu",
                "gpuType": "A100",
                "pricePerHour": 1.80,
                "memoryInGb": 80,
                "gpuCount": 1,
                "disk": 200,
                "region": "eu-central-1"
            }
        ]
    }
    mock_req.return_value = mock_resp

    sp = SpheronProvider(api_key="fake-key")
    offers = sp.list_gpus("4090", expand=True)
    self.assertEqual(len(offers), 1)
    self.assertEqual(offers[0]["id"], "spheron-4090-us")

  def test_spheron_rejects_template_id(self):
    test_args = ["cli.py", "create", "--provider", "spheron", "--template-id", "some-template"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  def test_spheron_rejects_cloud_type(self):
    test_args = ["cli.py", "create", "--provider", "spheron", "--cloud-type", "SECURE"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  @patch("xpu_shell.cli.resolve_provider")
  @patch("sys.stdout")
  def test_spheron_cli_gpus(self, mock_stdout, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "spheron"
    mock_prov.list_gpus.return_value = [
        {
            "id": "RTX 4090",
            "displayName": "RTX 4090",
            "memoryInGb": 24,
            "minPrice": 0.35,
            "maxPrice": 0.49,
            "count": 5
        }
    ]
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "--provider", "spheron", "--api-key", "fake-key", "gpus"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("RTX 4090", output)
    self.assertIn("$0.35/hr", output)
    self.assertIn("5 offers", output)

  @patch("xpu_shell.providers.spheron_provider.SpheronProvider.create_instance")
  @patch("xpu_shell.providers.spheron_provider.SpheronProvider.get_instance")
  @patch("xpu_shell.cli.run_container_setup")
  @patch("xpu_shell.cli.wait_for_ssh")
  @patch("xpu_shell.cli.find_ssh_private_key")
  @patch("xpu_shell.cli.get_ssh_key")
  @patch("time.sleep")
  def test_create_with_ssh_key_id_auto_detects_spheron(self, mock_sleep, mock_ssh_key, mock_find_priv, mock_wait_ssh, mock_setup, mock_get_inst, mock_create_inst):
    mock_ssh_key.return_value = "ssh-rsa fake_public_key"
    fake_inst = Instance(
        id="spheron-auto-1",
        name="test-worker",
        provider="spheron",
        status="RUNNING",
        ssh_host="ssh.spheron.ai",
        ssh_port=22
    )
    mock_create_inst.return_value = fake_inst
    mock_get_inst.return_value = fake_inst

    test_args = ["cli.py", "create", "--ssh-key-id", "spheron-key-999"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_create_inst.assert_called_once()
    self.assertEqual(mock_create_inst.call_args[1]["ssh_key_id"], "spheron-key-999")
    self.assertIsNone(mock_create_inst.call_args[1]["image_name"])

