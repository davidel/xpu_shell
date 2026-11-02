import os
import re
from typing import Any, Dict, List, Optional, Tuple
import runpod

from xpu_shell.providers.base import (
    BaseProvider,
    Instance,
    Template,
    format_template_identifier,
    parse_template_identifier
)


def get_pod_ssh_endpoint(pod_info: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
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

  if ssh_port is not None:
    try:
      ssh_port = int(ssh_port)
    except (ValueError, TypeError):
      pass

  return ssh_host, ssh_port


def normalize_runpod_instance(pod: Dict[str, Any]) -> Instance:
  ssh_host, ssh_port = get_pod_ssh_endpoint(pod)
  status = pod.get("desiredStatus") or pod.get("status") or "UNKNOWN"
  machine = pod.get("machine") or {}
  gpu_name = machine.get("gpuDisplayName") or pod.get("gpuType")
  gpu_count = pod.get("gpuCount") or machine.get("gpuCount") or 1
  cost = pod.get("costPerHr")
  vcpu = pod.get("vcpuCount")
  mem = pod.get("memoryInGb")
  disk = pod.get("containerDiskInGb") or pod.get("volumeInGb")

  return Instance(
      id=str(pod.get("id")),
      name=pod.get("name") or str(pod.get("id")),
      provider="runpod",
      status=status,
      ssh_host=ssh_host,
      ssh_port=ssh_port,
      gpu_name=gpu_name,
      gpu_count=int(gpu_count),
      cost_per_hr=float(cost) if cost is not None else None,
      vcpu_count=int(vcpu) if vcpu is not None else None,
      memory_gb=float(mem) if mem is not None else None,
      disk_gb=float(disk) if disk is not None else None,
      image_name=pod.get("imageName"),
      raw=pod
  )


class RunPodProvider(BaseProvider):

  def __init__(self, api_key: Optional[str] = None):
    self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")
    if self._api_key:
      runpod.api_key = self._api_key

  @property
  def name(self) -> str:
    return "runpod"

  def register_create_args(self, parser: Any) -> None:
    parser.add_argument(
        "--template-id",
        "--template_id",
        dest="template_id",
        default=None,
        help="RunPod template ID (takes precedence over --image-name)"
    )
    parser.add_argument(
        "--volume-id",
        help="Network volume ID to attach (optional)"
    )
    parser.add_argument(
        "--ports",
        default="22/tcp",
        help="Container ports to expose (default: %(default)s)"
    )
    parser.add_argument(
        "--docker-args",
        "--docker_args",
        dest="docker_args",
        default=None,
        help="Custom docker arguments to override container entrypoint (optional)"
    )
    parser.add_argument(
        "--cloud-type",
        choices=["SECURE", "COMMUNITY", "ALL"],
        default="SECURE",
        help="Type of cloud network to deploy the pod on (default: %(default)s)"
    )
    parser.add_argument(
        "--vcpu-count",
        "--vcpu_count",
        type=int,
        default=4,
        dest="vcpu_count",
        help="Minimum number of vCPUs to allocate (default: %(default)s)"
    )
    parser.add_argument(
        "--memory",
        type=int,
        default=8,
        help="Minimum CPU RAM in GB to allocate (default: %(default)s)"
    )

  def register_commands(self, subparsers: Any) -> None:
    pass

  def list_instances(self) -> List[Instance]:
    pods = runpod.get_pods() or []
    return [normalize_runpod_instance(p) for p in pods]

  def get_instance(self, instance_id: str) -> Instance:
    pod = runpod.get_pod(instance_id)
    if not pod:
      raise FileNotFoundError(f"RunPod instance '{instance_id}' not found.")
    return normalize_runpod_instance(pod)

  def get_ssh_endpoint(self, instance_id: str) -> Tuple[Optional[str], Optional[int]]:
    instance = self.get_instance(instance_id)
    return instance.ssh_host, instance.ssh_port

  def create_instance(self, **kwargs) -> Instance:
    if "template" in kwargs and kwargs["template"]:
      kind, val = parse_template_identifier(kwargs.pop("template"))
      if kind == "image":
        kwargs["image_name"] = val
      elif kind == "template":
        kwargs["template_id"] = val
    pod = runpod.create_pod(**kwargs)
    if not pod or not pod.get("id"):
      raise RuntimeError(f"RunPod failed to return a valid instance ID: {pod}")
    return self.get_instance(pod["id"])

  def stop_instance(self, instance_id: str) -> None:
    runpod.stop_pod(instance_id)

  def start_instance(self, instance_id: str) -> None:
    if hasattr(runpod, "resume_pod"):
      runpod.resume_pod(instance_id)
    elif hasattr(runpod, "start_pod"):
      runpod.start_pod(instance_id)

  def terminate_instance(self, instance_id: str) -> None:
    runpod.terminate_pod(instance_id)

  def list_gpus(self, filter_pattern: Optional[str] = None, expand: bool = False) -> List[Dict[str, Any]]:
    gpus = runpod.get_gpus() or []
    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      gpus = [g for g in gpus if regex.search(g.get("id", "")) or regex.search(g.get("displayName", ""))]
    return gpus

  def list_templates(self, filter_pattern: Optional[str] = None) -> List[Template]:
    templates: List[Template] = []
    try:
      from xpu_shell.cli import get_pod_templates
      raw_tpls = get_pod_templates()
    except Exception:
      try:
        from runpod.api.graphql import run_graphql_query
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
        resp = run_graphql_query(query)
        raw_tpls = resp.get("data", {}).get("myself", {}).get("podTemplates", [])
      except Exception:
        raw_tpls = []

    for t in raw_tpls:
      t_id = str(t.get("id") or "")
      name = t.get("name") or t_id
      img = t.get("imageName")
      templates.append(Template(
          id=format_template_identifier("template", t_id),
          name=name,
          provider="runpod",
          type="template",
          image_name=img,
          raw=t
      ))

    common_images = [
        ("runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04", "RunPod PyTorch 2.1 (CUDA 11.8)"),
        ("runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04", "RunPod PyTorch 2.2 (CUDA 12.1)"),
        ("runpod/tensorflow:2.14.0-py3.10-cuda11.8.0-devel-ubuntu22.04", "RunPod TensorFlow 2.14"),
        ("runpod/stable-diffusion:webui-11.8.0", "RunPod Stable Diffusion WebUI"),
    ]
    for img_name, desc in common_images:
      templates.append(Template(
          id=format_template_identifier("image", img_name),
          name=desc,
          provider="runpod",
          type="image",
          image_name=img_name
      ))

    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      templates = [
          t for t in templates
          if regex.search(t.id) or regex.search(t.name) or regex.search(t.image_name or "")
      ]

    return templates
