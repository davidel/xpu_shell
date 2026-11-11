import json
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
    parse_template_identifier,
    format_template_identifier,
    save_last_provider,
    get_last_provider
)
from xpu_shell.providers.base import Instance, Template
from xpu_shell.providers.runpod_provider import RunPodProvider
from xpu_shell.providers.vast_provider import VastProvider
from xpu_shell.providers.spheron_provider import SpheronProvider
from xpu_shell.providers.cloudrift_provider import CloudRiftProvider
from xpu_shell.providers.lium_provider import (
    LiumProvider,
    find_lium_api_key,
    normalize_lium_status,
    extract_lium_ssh_endpoint,
    normalize_lium_instance
)


class BaseProviderTestCase(unittest.TestCase):

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


class TestProviders(BaseProviderTestCase):

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

    prov, pid = parse_instance_identifier("cloudrift:cr-999")
    self.assertEqual(prov, "cloudrift")
    self.assertEqual(pid, "cr-999")

    prov, pid = parse_instance_identifier("cr:cr-888")
    self.assertEqual(prov, "cr")
    self.assertEqual(pid, "cr-888")

    prov, pid = parse_instance_identifier("lium:pod-uuid-123")
    self.assertEqual(prov, "lium")
    self.assertEqual(pid, "pod-uuid-123")

    prov, pid = parse_instance_identifier("plain-id")
    self.assertIsNone(prov)
    self.assertEqual(pid, "plain-id")

  def test_parse_and_format_template_identifier(self):
    self.assertEqual(format_template_identifier("image", "pytorch/pytorch:latest"), "image:pytorch/pytorch:latest")
    self.assertEqual(format_template_identifier("template", "c21f8a"), "template:c21f8a")
    self.assertEqual(format_template_identifier("image", "image:already-prefixed"), "image:already-prefixed")

    kind, val = parse_template_identifier("image:pytorch/pytorch:latest")
    self.assertEqual(kind, "image")
    self.assertEqual(val, "pytorch/pytorch:latest")

    kind, val = parse_template_identifier("template:c21f8a")
    self.assertEqual(kind, "template")
    self.assertEqual(val, "c21f8a")

    kind, val = parse_template_identifier("pytorch/pytorch:latest")
    self.assertEqual(kind, "image")
    self.assertEqual(val, "pytorch/pytorch:latest")

    kind, val = parse_template_identifier("c21f8a")
    self.assertEqual(kind, "template")
    self.assertEqual(val, "c21f8a")

    kind, val = parse_template_identifier("Ubuntu 22.04 LTS")
    self.assertEqual(kind, "image")
    self.assertEqual(val, "Ubuntu 22.04 LTS")

    self.assertEqual(parse_template_identifier(None), (None, None))

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

    cr = get_provider("cloudrift", api_key="test-key")
    self.assertIsInstance(cr, CloudRiftProvider)
    self.assertEqual(cr.name, "cloudrift")

    cr_alias = get_provider("cr", api_key="test-key")
    self.assertIsInstance(cr_alias, CloudRiftProvider)

    lium = get_provider("lium", api_key="test-key")
    self.assertIsInstance(lium, LiumProvider)
    self.assertEqual(lium.name, "lium")

    lium_alias = get_provider("lium.io", api_key="test-key")
    self.assertIsInstance(lium_alias, LiumProvider)

    with self.assertRaises(ValueError):
      get_provider("invalid-provider")

  def test_resolve_provider_precedence(self):
    # 1. Explicit CLI argument
    args = MagicMock(provider="vast", api_key="vast-key")
    p = resolve_provider(args=args)
    self.assertIsInstance(p, VastProvider)

    args = MagicMock(provider="cloudrift", api_key="cr-key")
    p = resolve_provider(args=args)
    self.assertIsInstance(p, CloudRiftProvider)

    args = MagicMock(provider="lium", api_key="lium-key")
    p = resolve_provider(args=args)
    self.assertIsInstance(p, LiumProvider)

    args = MagicMock(provider=None, datacenter="us-east-1", ssh_key_id=None, offer_id=None, executor_id=None, api_key="cr-key")
    p = resolve_provider(args=args)
    self.assertIsInstance(p, CloudRiftProvider)

    args = MagicMock(provider=None, datacenter=None, ssh_key_id=None, offer_id=None, executor_id="exec-123", api_key="lium-key")
    p = resolve_provider(args=args)
    self.assertIsInstance(p, LiumProvider)

    # 2. Prefixed ID
    p = resolve_provider(instance_id="vast:12345")
    self.assertIsInstance(p, VastProvider)

    p = resolve_provider(instance_id="spheron:dep-111")
    self.assertIsInstance(p, SpheronProvider)

    p = resolve_provider(instance_id="cloudrift:cr-555")
    self.assertIsInstance(p, CloudRiftProvider)

    p = resolve_provider(instance_id="lium:pod-999")
    self.assertIsInstance(p, LiumProvider)

    p = resolve_provider(instance_id="cr-node-222")
    self.assertIsInstance(p, CloudRiftProvider)

    p = resolve_provider(instance_id="spheron-node-222")
    self.assertIsInstance(p, SpheronProvider)

    p = resolve_provider(instance_id="lium-node-333")
    self.assertIsInstance(p, LiumProvider)

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


class TestRunPodProvider(BaseProviderTestCase):

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


class TestVastProvider(BaseProviderTestCase):

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
    called_params = mock_req.call_args[1].get("params", {})
    self.assertIn("q", called_params)
    parsed_q = json.loads(called_params["q"])
    self.assertEqual(parsed_q.get("rentable"), {"eq": True})
    self.assertEqual(parsed_q.get("rented"), {"eq": False})

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

  @patch("requests.Session.request")
  def test_create_instance_with_template(self, mock_req):
    mock_put_resp = MagicMock(status_code=200, content=b'{}', json=MagicMock(return_value={"new_contract": 6666}))
    mock_get_resp = MagicMock(status_code=200, content=b'{}', json=MagicMock(return_value={"instances": [{"id": 6666, "actual_status": "running"}]}))
    mock_req.side_effect = [mock_put_resp, mock_get_resp]

    vast = VastProvider(api_key="fake-vast-key")
    # Using template:hash
    vast.create_instance(offer_id=1234, template="template:tpl-hash-123")
    body = mock_req.call_args_list[0][1]["json"]
    self.assertEqual(body.get("template_hash_id"), "tpl-hash-123")

    # Using image:custom/image:tag
    mock_req.reset_mock()
    mock_req.side_effect = [mock_put_resp, mock_get_resp]
    vast.create_instance(offer_id=1234, template="image:custom/pytorch:latest")
    body = mock_req.call_args_list[0][1]["json"]
    self.assertEqual(body.get("image"), "custom/pytorch:latest")

  @patch("requests.Session.request")
  def test_list_templates(self, mock_req):
    mock_resp = MagicMock(status_code=200, content=b'{}')
    mock_resp.json.return_value = {
        "templates": [
            {
                "hash_id": "abc123hash",
                "name": "PyTorch Dev Template",
                "image": "pytorch/pytorch:2.1",
                "desc": "A great dev template"
            }
        ]
    }
    mock_req.return_value = mock_resp

    vast = VastProvider(api_key="fake-vast-key")
    tpls = vast.list_templates()
    self.assertTrue(any(t.id == "template:abc123hash" for t in tpls))
    self.assertTrue(any(t.id == "image:pytorch/pytorch:latest" for t in tpls))


class TestVastCLIIntegration(BaseProviderTestCase):

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

    test_args = ["cli.py", "list", "--provider", "vast", "--api-key", "fake-key"]
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

    test_args = ["cli.py", "stop", "--provider", "vast", "--api-key", "fake-key", "--pod", "12345"]
    with patch.object(sys, "argv", test_args):
      cli.main()

    mock_prov.stop_instance.assert_called_once_with("12345")

  @patch("xpu_shell.cli.resolve_provider")
  def test_vast_cli_terminate(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "terminate", "--provider", "vast", "--api-key", "fake-key", "--pod", "12345"]
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
        "exec",
        "--provider", "vast",
        "--api-key", "fake-key",
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
  def test_vast_cli_info(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_prov.get_instance.return_value = Instance(
        id="58842137",
        name="vast-58842137",
        provider="vast",
        status="RUNNING",
        ssh_host="ssh2.vast.ai",
        ssh_port=22516,
        gpu_name="RTX 4090",
        gpu_count=1,
        uptime_seconds=3600,
        cost_per_hr=0.54,
        vcpu_count=32,
        memory_gb=108.0,
        disk_gb=80.0,
        image_name="pytorch/pytorch:latest",
        raw={
            "id": 58842137,
            "actual_status": "running",
            "geolocation": "California, US",
            "cpu_name": "AMD EPYC 9684X",
            "cpu_util": 5.0,
            "mem_usage": 10.0,
            "gpu_util": 80.0,
            "vmem_usage": 45.0
        }
    )
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "info", "--provider", "vast", "--api-key", "fake-key", "-p", "58842137"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_prov.get_instance.assert_called_once_with("58842137")
    output = "\n".join(call[0][0] for call in mock_print.call_args_list if call[0])
    self.assertIn("vast-58842137 (58842137)", output)
    self.assertIn("RUNNING", output)
    self.assertIn("1x RTX 4090", output)
    self.assertIn("ssh2.vast.ai:22516", output)
    self.assertIn("California, US", output)

  @patch("xpu_shell.cli.resolve_provider")
  def test_vast_cli_info_json(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "vast"
    mock_prov.get_instance.return_value = Instance(
        id="58842137",
        name="vast-58842137",
        provider="vast",
        status="RUNNING",
        raw={"id": 58842137, "status": "running"}
    )
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "info", "--provider", "vast", "--api-key", "fake-key", "--json", "58842137"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_prov.get_instance.assert_called_once_with("58842137")
    mock_print.assert_called_once()
    self.assertIn('"id": 58842137', mock_print.call_args[0][0])

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

    test_args = ["cli.py", "gpus", "--provider", "vast", "--api-key", "fake-key"]
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

    test_args = ["cli.py", "gpus", "--provider", "vast", "--api-key", "fake-key", "-l", "4090"]
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

  def test_rejects_global_provider(self):
    test_args = ["cli.py", "--provider", "runpod", "create"]
    with patch.object(sys, "argv", test_args):
      with patch("sys.stderr"):
        with self.assertRaises(SystemExit) as ctx:
          cli.main()
        self.assertEqual(ctx.exception.code, 2)

  @patch("sys.stdout")
  def test_vast_cli_templates(self, mock_stdout):
    test_args = ["cli.py", "templates", "--provider", "vast", "--api-key", "fake-key"]
    with patch.object(sys, "argv", test_args):
      cli.main()
    calls = [call[0][0] for call in mock_stdout.write.call_args_list if call[0]]
    output = "".join(calls)
    self.assertIn("image:pytorch/pytorch:latest", output)

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


class TestSpheronProvider(BaseProviderTestCase):

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
  def test_create_instance_with_template(self, mock_req):
    mock_post_resp = MagicMock(status_code=200, content=b'{}', json=MagicMock(return_value={"id": "dep-tpl"}))
    mock_get_resp = MagicMock(status_code=200, content=b'{}', json=MagicMock(return_value={"id": "dep-tpl", "status": "Running"}))
    mock_req.side_effect = [mock_post_resp, mock_get_resp]

    sp = SpheronProvider(api_key="fake-key")
    inst = sp.create_instance(offer_id="off-1", template="image:debian-12")
    self.assertEqual(inst.id, "dep-tpl")
    body = mock_req.call_args_list[0][1]["json"]
    self.assertEqual(body["operatingSystem"], "debian-12")

  def test_list_templates(self):
    sp = SpheronProvider(api_key="fake-key")
    tpls = sp.list_templates()
    self.assertTrue(any(t.id == "image:ubuntu-22.04" for t in tpls))
    self.assertTrue(any(t.id == "image:debian-12" for t in tpls))

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

  @patch("requests.Session.request")
  def test_list_gpus_grouped_structure(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{}'
    mock_resp.json.return_value = {
        "page": 1,
        "totalPages": 1,
        "total": 2,
        "data": [
            {
                "gpuType": "B300_SXM6",
                "displayName": "B300 SXM6",
                "offers": [
                    {
                        "offerId": "spheron-b300-1",
                        "price": 6.67,
                        "gpu_memory": 262,
                        "gpuCount": 1,
                        "storage": 256,
                        "available": True,
                        "clusters": ["UK South 1"]
                    },
                    {
                        "offerId": "spheron-b300-2",
                        "price": 9.21,
                        "gpu_memory": 262,
                        "gpuCount": 1,
                        "storage": 256,
                        "available": True,
                        "clusters": ["EU West 2"]
                    }
                ]
            },
            {
                "gpuType": "CPU",
                "displayName": "CPU NODE",
                "offers": [
                    {
                        "offerId": "spheron-cpu-1",
                        "price": 0.09,
                        "gpu_memory": 0,
                        "gpuCount": 1,
                        "storage": 100,
                        "available": True
                    }
                ]
            }
        ]
    }
    mock_req.return_value = mock_resp

    sp = SpheronProvider(api_key="fake-key")
    gpus = sp.list_gpus()
    self.assertEqual(len(gpus), 2)
    b300 = next(g for g in gpus if g["displayName"] == "B300_SXM6")
    self.assertEqual(b300["count"], 2)
    self.assertEqual(b300["minPrice"], 6.67)
    self.assertEqual(b300["maxPrice"], 9.21)
    self.assertEqual(b300["memoryInGb"], 262)

    cpu = next(g for g in gpus if g["displayName"] == "CPU")
    self.assertEqual(cpu["count"], 1)
    self.assertEqual(cpu["minPrice"], 0.09)
    self.assertEqual(cpu["memoryInGb"], 0)

    expand_offers = sp.list_gpus("B300", expand=True)
    self.assertEqual(len(expand_offers), 2)
    self.assertEqual(expand_offers[0]["id"], "spheron-b300-1")
    self.assertEqual(expand_offers[0]["memoryInGb"], 262)
    self.assertEqual(expand_offers[0]["disk"], 256)
    self.assertEqual(expand_offers[0]["region"], "UK South 1")

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

    test_args = ["cli.py", "gpus", "--provider", "spheron", "--api-key", "fake-key"]
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

  @patch("xpu_shell.cli.resolve_provider")
  def test_spheron_cli_info(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "spheron"
    mock_prov.get_instance.return_value = Instance(
        id="spheron-auto-1",
        name="test-worker",
        provider="spheron",
        status="RUNNING",
        ssh_host="ssh.spheron.ai",
        ssh_port=22,
        gpu_name="RTX 4090",
        gpu_count=1,
        uptime_seconds=1800,
        cost_per_hr=0.42,
        vcpu_count=8,
        memory_gb=32.0,
        disk_gb=100.0,
        image_name="ubuntu:22.04",
        raw={"region": "us-east-1"}
    )
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "info", "--provider", "spheron", "--api-key", "fake-key", "-p", "spheron-auto-1"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_prov.get_instance.assert_called_once_with("spheron-auto-1")
    output = "\n".join(call[0][0] for call in mock_print.call_args_list if call[0])
    self.assertIn("test-worker (spheron-auto-1)", output)
    self.assertIn("RUNNING", output)
    self.assertIn("1x RTX 4090", output)
    self.assertIn("ssh.spheron.ai:22", output)
    self.assertIn("us-east-1", output)

  @patch("xpu_shell.cli.resolve_provider")
  def test_spheron_cli_info_json(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "spheron"
    mock_prov.get_instance.return_value = Instance(
        id="spheron-auto-1",
        name="test-worker",
        provider="spheron",
        status="RUNNING",
        raw={"deploymentId": "spheron-auto-1", "state": "running"}
    )
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "info", "--provider", "spheron", "--api-key", "fake-key", "--json", "spheron-auto-1"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_prov.get_instance.assert_called_once_with("spheron-auto-1")
    mock_print.assert_called_once()
    self.assertIn('"deploymentId": "spheron-auto-1"', mock_print.call_args[0][0])


class TestCloudRiftProvider(BaseProviderTestCase):

  @patch("requests.Session.request")
  def test_list_instances(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"data": {"instances": []}}'
    mock_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "instances": [
                {
                    "id": "cr-inst-123",
                    "instance_name": "my-cr-worker",
                    "status": "Active",
                    "host_address": "198.51.100.10",
                    "gpus": [
                        {"brand_short": "RTX 4090"}
                    ],
                    "cost_per_hour": 50,
                    "cpu_limit": 16,
                    "dram_limit": 68719476736,
                    "disk_limit": 107374182400,
                    "virtual_machines": [{"name": "ubuntu"}]
                }
            ]
        }
    }
    mock_req.return_value = mock_resp

    provider = CloudRiftProvider(api_key="test-cr-key")
    instances = provider.list_instances()
    self.assertEqual(len(instances), 1)
    inst = instances[0]
    self.assertEqual(inst.id, "cr-inst-123")
    self.assertEqual(inst.name, "my-cr-worker")
    self.assertEqual(inst.status, "RUNNING")
    self.assertEqual(inst.ssh_host, "198.51.100.10")
    self.assertEqual(inst.ssh_port, 22)
    self.assertEqual(inst.gpu_name, "RTX 4090")
    self.assertEqual(inst.gpu_count, 1)
    self.assertEqual(inst.cost_per_hr, 0.50)
    self.assertEqual(inst.vcpu_count, 16)
    self.assertAlmostEqual(inst.memory_gb, 64.0)
    self.assertAlmostEqual(inst.disk_gb, 100.0)

  @patch("requests.Session.request")
  def test_get_instance_and_ssh_endpoint(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"data": {"instances": []}}'
    mock_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "instances": [
                {
                    "id": "cr-inst-456",
                    "instance_name": "worker-456",
                    "status": "Active",
                    "host_address": "203.0.113.5",
                    "port_mappings": [[2222, 22]]
                }
            ]
        }
    }
    mock_req.return_value = mock_resp

    provider = CloudRiftProvider(api_key="test-cr-key")
    inst = provider.get_instance("cr-inst-456")
    self.assertEqual(inst.id, "cr-inst-456")
    self.assertEqual(inst.ssh_host, "203.0.113.5")
    self.assertEqual(inst.ssh_port, 2222)

    host, port = provider.get_ssh_endpoint("cr-inst-456")
    self.assertEqual(host, "203.0.113.5")
    self.assertEqual(port, 2222)

  @patch("requests.Session.request")
  def test_create_instance_vm(self, mock_req):
    mock_types_resp = MagicMock()
    mock_types_resp.status_code = 200
    mock_types_resp.content = b'{"data": {}}'
    mock_types_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "instance_types": [
                {
                    "name": "rtx4090",
                    "brand_short": "RTX 4090",
                    "cost_per_hour": 45,
                    "datacenters": [{"name": "us-east-1"}],
                    "variants": [
                        {
                            "cost_per_hour": 45,
                            "gpu_count": 1,
                            "dram": 34359738368,
                            "disk": 107374182400,
                            "available_nodes": 2
                        }
                    ]
                }
            ]
        }
    }

    mock_rent_resp = MagicMock()
    mock_rent_resp.status_code = 200
    mock_rent_resp.content = b'{"data": {}}'
    mock_rent_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "instance_ids": ["cr-inst-new-1"]
        }
    }

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{"data": {}}'
    mock_get_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "instances": [
                {
                    "id": "cr-inst-new-1",
                    "name": "cloudrift-worker",
                    "status": "Initializing",
                    "host_address": "198.51.100.99"
                }
            ]
        }
    }

    mock_req.side_effect = [mock_types_resp, mock_rent_resp, mock_get_resp]

    provider = CloudRiftProvider(api_key="test-cr-key")
    inst = provider.create_instance(
        name="cloudrift-worker",
        gpu_name="RTX 4090",
        ssh_public_key="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC...",
        datacenter="us-east-1"
    )

    self.assertEqual(inst.id, "cr-inst-new-1")
    self.assertEqual(inst.status, "INITIALIZING")

    rent_call = mock_req.call_args_list[1]
    rent_payload = rent_call[1]["json"]
    self.assertEqual(rent_payload["version"], "~upcoming")
    self.assertEqual(rent_payload["data"]["selector"]["ByInstanceTypeAndLocation"]["instance_type"], "rtx4090")
    self.assertEqual(rent_payload["data"]["selector"]["ByInstanceTypeAndLocation"]["datacenters"], ["us-east-1"])
    self.assertIn("VirtualMachine", rent_payload["data"]["config"])
    self.assertIn("PublicKeys", rent_payload["data"]["config"]["VirtualMachine"]["ssh_key"])

  @patch("requests.Session.request")
  def test_terminate_instance(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"data": {}}'
    mock_resp.json.return_value = {"version": "~upcoming", "data": {}}
    mock_req.return_value = mock_resp

    provider = CloudRiftProvider(api_key="test-cr-key")
    provider.terminate_instance("cr-inst-123")
    self.assertEqual(mock_req.call_count, 1)
    call_json = mock_req.call_args[1]["json"]
    self.assertEqual(call_json["data"]["selector"]["ById"], ["cr-inst-123"])

  @patch("requests.Session.request")
  def test_list_gpus(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"data": {}}'
    mock_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "instance_types": [
                {
                    "name": "rtx4090",
                    "brand_short": "RTX 4090",
                    "cost_per_hour": 50,
                    "variants": [
                        {
                            "cost_per_hour": 50,
                            "gpu_count": 1,
                            "dram": 25769803776,
                            "disk": 107374182400,
                            "available_nodes": 5
                        }
                    ]
                }
            ]
        }
    }
    mock_req.return_value = mock_resp

    provider = CloudRiftProvider(api_key="test-cr-key")
    gpus = provider.list_gpus()
    self.assertEqual(len(gpus), 1)
    self.assertEqual(gpus[0]["id"], "rtx4090")
    self.assertEqual(gpus[0]["displayName"], "RTX 4090")
    self.assertEqual(gpus[0]["minPrice"], 0.50)

  @patch("requests.Session.request")
  def test_list_templates(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"data": {}}'
    mock_resp.json.return_value = {
        "version": "~upcoming",
        "data": {
            "groups": [
                {
                    "recipes": [
                        {"name": "ubuntu-22.04"}
                    ]
                }
            ]
        }
    }
    mock_req.return_value = mock_resp

    provider = CloudRiftProvider(api_key="test-cr-key")
    templates = provider.list_templates()
    self.assertEqual(len(templates), 1)
    self.assertEqual(templates[0].id, "template:ubuntu-22.04")
    self.assertEqual(templates[0].provider, "cloudrift")

  @patch("xpu_shell.cli.resolve_provider")
  def test_cloudrift_cli_info(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "cloudrift"
    mock_prov.get_instance.return_value = Instance(
        id="cr-auto-1",
        name="test-cr-node",
        provider="cloudrift",
        status="RUNNING",
        ssh_host="198.51.100.22",
        ssh_port=22,
        gpu_name="RTX 4090",
        gpu_count=1,
        uptime_seconds=3600,
        cost_per_hr=0.50,
        vcpu_count=16,
        memory_gb=64.0,
        disk_gb=100.0,
        image_name="ubuntu",
        raw={"datacenter": "us-east-1"}
    )
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "info", "--provider", "cloudrift", "--api-key", "fake-cr-key", "-p", "cr-auto-1"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_prov.get_instance.assert_called_once_with("cr-auto-1")
    output = "\n".join(call[0][0] for call in mock_print.call_args_list if call[0])
    self.assertIn("test-cr-node (cr-auto-1)", output)
    self.assertIn("RUNNING", output)
    self.assertIn("1x RTX 4090", output)
    self.assertIn("198.51.100.22:22", output)

  def test_lium_find_api_key(self):
    self.assertEqual(find_lium_api_key("explicit-key"), "explicit-key")
    with patch.dict(os.environ, {"LIUM_API_KEY": "env-key"}):
      self.assertEqual(find_lium_api_key(), "env-key")

    with patch.dict(os.environ, {}, clear=True):
      with patch("pathlib.Path.is_file", return_value=True):
        with patch("pathlib.Path.read_text", return_value="file-key"):
          self.assertEqual(find_lium_api_key(), "file-key")

  def test_lium_normalize_status(self):
    self.assertEqual(normalize_lium_status("running"), "RUNNING")
    self.assertEqual(normalize_lium_status("active"), "RUNNING")
    self.assertEqual(normalize_lium_status("pending"), "INITIALIZING")
    self.assertEqual(normalize_lium_status("start_pending"), "INITIALIZING")
    self.assertEqual(normalize_lium_status("stopped"), "STOPPED")
    self.assertEqual(normalize_lium_status("deleting"), "TERMINATED")
    self.assertEqual(normalize_lium_status("failed"), "FAILED")
    self.assertEqual(normalize_lium_status("creation_failed"), "FAILED")
    self.assertEqual(normalize_lium_status(None), "UNKNOWN")

  def test_lium_extract_ssh_endpoint(self):
    h, p = extract_lium_ssh_endpoint({"ssh_connect_cmd": "ssh root@192.0.2.10 -p 22001"})
    self.assertEqual(h, "192.0.2.10")
    self.assertEqual(p, 22001)

    h, p = extract_lium_ssh_endpoint({"ssh_connect_cmd": "ssh -p 33333 user@gpu.lium.io"})
    self.assertEqual(h, "gpu.lium.io")
    self.assertEqual(p, 33333)

    h, p = extract_lium_ssh_endpoint({
        "ssh_connect_cmd": "",
        "executor_ip_address": "198.51.100.5",
        "ports_mapping": {"22/tcp": 22022}
    })
    self.assertEqual(h, "198.51.100.5")
    self.assertEqual(p, 22022)

  def test_lium_normalize_instance(self):
    raw = {
        "id": "lium-pod-101",
        "pod_name": "my-lium-pod",
        "status": "RUNNING",
        "ssh_connect_cmd": "ssh root@10.20.30.40 -p 22050",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_count": 2,
        "price": 3.50,
        "cpu_count": 32,
        "ram_total": 128 * 1024 * 1024 * 1024,
        "storage_limit_gb": 200,
        "docker_image": "daturaai/pytorch:latest"
    }
    inst = normalize_lium_instance(raw)
    self.assertEqual(inst.id, "lium-pod-101")
    self.assertEqual(inst.name, "my-lium-pod")
    self.assertEqual(inst.provider, "lium")
    self.assertEqual(inst.status, "RUNNING")
    self.assertEqual(inst.ssh_host, "10.20.30.40")
    self.assertEqual(inst.ssh_port, 22050)
    self.assertEqual(inst.gpu_name, "NVIDIA H100 80GB HBM3")
    self.assertEqual(inst.gpu_count, 2)
    self.assertEqual(inst.cost_per_hr, 3.50)
    self.assertEqual(inst.vcpu_count, 32)
    self.assertEqual(inst.memory_gb, 128.0)
    self.assertEqual(inst.disk_gb, 200.0)
    self.assertEqual(inst.image_name, "daturaai/pytorch:latest")

  @patch("requests.Session.request")
  def test_lium_list_and_get_instances(self, mock_req):
    mock_list_resp = MagicMock()
    mock_list_resp.status_code = 200
    mock_list_resp.content = b'[{"id": "pod-1", "pod_name": "p1", "status": "RUNNING"}]'
    mock_list_resp.json.return_value = [{"id": "pod-1", "pod_name": "p1", "status": "RUNNING"}]

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{"id": "pod-1", "pod_name": "p1", "status": "RUNNING", "ssh_connect_cmd": "ssh root@1.2.3.4 -p 22"}'
    mock_get_resp.json.return_value = {
        "id": "pod-1",
        "pod_name": "p1",
        "status": "RUNNING",
        "ssh_connect_cmd": "ssh root@1.2.3.4 -p 22"
    }

    mock_req.side_effect = [mock_list_resp, mock_get_resp, mock_get_resp]

    provider = LiumProvider(api_key="test-lium-key")
    pods = provider.list_instances()
    self.assertEqual(len(pods), 1)
    self.assertEqual(pods[0].id, "pod-1")
    self.assertEqual(pods[0].provider, "lium")

    single = provider.get_instance("pod-1")
    self.assertEqual(single.id, "pod-1")
    self.assertEqual(single.status, "RUNNING")

    host, port = provider.get_ssh_endpoint("pod-1")
    self.assertEqual(host, "1.2.3.4")
    self.assertEqual(port, 22)

  @patch("requests.Session.request")
  def test_lium_create_instance_spec(self, mock_req):
    mock_spec_resp = MagicMock()
    mock_spec_resp.status_code = 200
    mock_spec_resp.content = b'{"pod_id": "pod-new-uuid", "price_per_hour": 1.25}'
    mock_spec_resp.json.return_value = {
        "pod_id": "pod-new-uuid",
        "price_per_hour": 1.25
    }

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{"id": "pod-new-uuid", "pod_name": "worker", "status": "PENDING"}'
    mock_get_resp.json.return_value = {
        "id": "pod-new-uuid",
        "pod_name": "worker",
        "status": "PENDING"
    }

    mock_req.side_effect = [mock_spec_resp, mock_get_resp]

    provider = LiumProvider(api_key="test-lium-key")
    inst = provider.create_instance(
        name="worker",
        gpu_name="RTX4090",
        ssh_public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...",
        gpu_count=1
    )
    self.assertEqual(inst.id, "pod-new-uuid")
    self.assertEqual(inst.status, "INITIALIZING")
    self.assertEqual(inst.provider, "lium")

  @patch("requests.Session.request")
  def test_lium_create_instance_executor(self, mock_req):
    mock_rent_resp = MagicMock()
    mock_rent_resp.status_code = 200
    mock_rent_resp.content = b'{"id": "pod-direct-uuid", "pod_name": "direct-worker", "status": "RUNNING"}'
    mock_rent_resp.json.return_value = {
        "id": "pod-direct-uuid",
        "pod_name": "direct-worker",
        "status": "RUNNING"
    }

    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.content = b'{"id": "pod-direct-uuid", "pod_name": "direct-worker", "status": "RUNNING"}'
    mock_get_resp.json.return_value = {
        "id": "pod-direct-uuid",
        "pod_name": "direct-worker",
        "status": "RUNNING"
    }

    mock_req.side_effect = [mock_rent_resp, mock_get_resp]

    provider = LiumProvider(api_key="test-lium-key")
    inst = provider.create_instance(
        name="direct-worker",
        executor_id="exec-uuid-999",
        ssh_public_key="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI..."
    )
    self.assertEqual(inst.id, "pod-direct-uuid")
    self.assertEqual(inst.provider, "lium")

  @patch("requests.Session.request")
  def test_lium_terminate_and_stop(self, mock_req):
    mock_del_resp = MagicMock()
    mock_del_resp.status_code = 200
    mock_del_resp.content = b'{"success": true}'
    mock_del_resp.json.return_value = {"success": True}
    mock_req.return_value = mock_del_resp

    provider = LiumProvider(api_key="test-lium-key")
    provider.stop_instance("pod-123")
    provider.terminate_instance("pod-123")

    with self.assertRaises(NotImplementedError):
      provider.start_instance("pod-123")

  @patch("requests.Session.request")
  def test_lium_list_gpus(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'[]'
    mock_resp.json.return_value = [
        {
            "id": "exec-1",
            "machine_name": "node-1",
            "price_per_gpu": 1.50,
            "gpu_count": 2,
            "reliability_score": 0.99,
            "specs": {
                "gpu": {
                    "count": 2,
                    "details": [
                        {"name": "NVIDIA H100 80GB HBM3", "capacity": 81559}
                    ]
                },
                "network": {
                    "ema_verifyx_download_speed": 1000,
                    "ema_verifyx_upload_speed": 500
                },
                "hard_disk": {
                    "total": 500 * (1024 ** 3)
                }
            }
        }
    ]
    mock_req.return_value = mock_resp

    provider = LiumProvider(api_key="test-lium-key")
    aggs = provider.list_gpus(expand=False)
    self.assertEqual(len(aggs), 1)
    self.assertEqual(aggs[0]["displayName"], "NVIDIA H100 80GB HBM3")
    self.assertEqual(aggs[0]["minPrice"], 3.0)

    offers = provider.list_gpus(expand=True)
    self.assertEqual(len(offers), 1)
    self.assertEqual(offers[0]["id"], "exec-1")
    self.assertEqual(offers[0]["num_gpus"], 2)

  @patch("requests.Session.request")
  def test_lium_list_templates(self, mock_req):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'[]'
    mock_resp.json.return_value = [
        {
            "id": "bb50bd64-e363-4adc-8111-1d44c8e36807",
            "name": "Pytorch (Cuda + DinD)",
            "description": "Standard PyTorch environment",
            "docker_image": "daturaai/pytorch",
            "docker_image_tag": "latest"
        }
    ]
    mock_req.return_value = mock_resp

    provider = LiumProvider(api_key="test-lium-key")
    templates = provider.list_templates()
    self.assertEqual(len(templates), 1)
    self.assertEqual(templates[0].id, "template:bb50bd64-e363-4adc-8111-1d44c8e36807")
    self.assertEqual(templates[0].image_name, "daturaai/pytorch:latest")
    self.assertEqual(templates[0].provider, "lium")

  @patch("xpu_shell.cli.resolve_provider")
  def test_lium_cli_info(self, mock_resolve_prov):
    mock_prov = MagicMock()
    mock_prov.name = "lium"
    mock_prov.get_instance.return_value = Instance(
        id="pod-cli-test",
        name="test-lium-pod",
        provider="lium",
        status="RUNNING",
        ssh_host="192.0.2.55",
        ssh_port=22022,
        gpu_name="NVIDIA H100 80GB HBM3",
        gpu_count=1,
        uptime_seconds=1800,
        cost_per_hr=1.50,
        vcpu_count=16,
        memory_gb=64.0,
        disk_gb=100.0,
        image_name="daturaai/pytorch:latest",
        raw={}
    )
    mock_resolve_prov.return_value = mock_prov

    test_args = ["cli.py", "info", "--provider", "lium", "--api-key", "fake-lium-key", "-p", "pod-cli-test"]
    with patch.object(sys, "argv", test_args):
      with patch("builtins.print") as mock_print:
        cli.main()

    mock_prov.get_instance.assert_called_once_with("pod-cli-test")
    output = "\n".join(call[0][0] for call in mock_print.call_args_list if call[0])
    self.assertIn("test-lium-pod (pod-cli-test)", output)
    self.assertIn("RUNNING", output)
    self.assertIn("1x NVIDIA H100 80GB HBM3", output)
    self.assertIn("192.0.2.55:22022", output)


