import os
import re
from typing import Any, Dict, List, Optional, Tuple
import runpod

from xpu_shell.providers.base import (
    BaseProvider,
    Instance,
    Template,
    format_template_identifier,
    parse_ssh_command,
    parse_template_identifier
)


def get_pod_ssh_endpoint(pod_info: Optional[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
  if not pod_info or not isinstance(pod_info, dict):
    return None, None

  ssh_host = None
  ssh_port = None

  ssh_block = pod_info.get("ssh")
  if isinstance(ssh_block, dict):
    direct = ssh_block.get("direct")
    if isinstance(direct, dict):
      ssh_host = direct.get("host")
      ssh_port = direct.get("port")
      if not ssh_host or not ssh_port:
        c_host, c_port = parse_ssh_command(direct.get("command"))
        ssh_host = ssh_host or c_host
        ssh_port = ssh_port or c_port

  if not ssh_host or not ssh_port:
    runtime = pod_info.get("runtime") or {}
    ports = runtime.get("ports") or []
    if isinstance(ports, list):
      for p in ports:
        if not isinstance(p, dict):
          continue
        priv = p.get("private") if p.get("private") is not None else p.get("privatePort")
        if priv == 22:
          pub = p.get("public") if p.get("public") is not None else p.get("publicPort")
          if pub is None:
            pub = p.get("isExternal")
          ssh_port = pub
          ssh_host = p.get("ip") or p.get("address")
          break

    if not ssh_host:
      ssh_host = pod_info.get("ipAddress") or pod_info.get("address")

    if not ssh_port and isinstance(ports, dict):
      ssh_port = ports.get("public") or ports.get("publicPort") or ports.get("isExternal")

  if not ssh_host or not ssh_port:
    ssh_block = pod_info.get("ssh")
    if isinstance(ssh_block, dict):
      proxy = ssh_block.get("proxy")
      if isinstance(proxy, dict):
        p_host = proxy.get("host")
        p_port = proxy.get("port")
        if not p_host or not p_port:
          c_host, c_port = parse_ssh_command(proxy.get("command"))
          p_host = p_host or c_host
          p_port = p_port or c_port
        ssh_host = ssh_host or p_host
        ssh_port = ssh_port or p_port

  if ssh_port is not None:
    try:
      ssh_port = int(ssh_port)
    except (ValueError, TypeError):
      ssh_port = None

  return ssh_host, ssh_port


def normalize_runpod_status(raw_status: Optional[str]) -> str:
  if not raw_status:
    return "UNKNOWN"
  st = raw_status.upper().strip()
  if st in ("RUNNING", "ACTIVE"):
    return "RUNNING"
  if st in ("CREATING", "INITIALIZING", "PENDING", "STARTING"):
    return "INITIALIZING"
  if st in ("STOPPED", "PAUSED", "EXITED"):
    return "STOPPED"
  if st in ("TERMINATED", "DELETED"):
    return "TERMINATED"
  if st in ("FAILED", "ERROR", "DEAD"):
    return "FAILED"
  return st


def normalize_runpod_instance(pod: Dict[str, Any]) -> Instance:
  ssh_host, ssh_port = get_pod_ssh_endpoint(pod)
  raw_status = pod.get("status") or pod.get("desiredStatus")
  status = normalize_runpod_status(raw_status)
  machine = pod.get("machine") or {}
  gpu_info = pod.get("gpu") if isinstance(pod.get("gpu"), dict) else {}
  cpu_info = pod.get("cpu") if isinstance(pod.get("cpu"), dict) else {}

  gpu_count = pod.get("gpuCount")
  if gpu_count is None and "count" in gpu_info:
    gpu_count = gpu_info.get("count")
  if gpu_count is None and machine.get("gpuCount") is not None:
    gpu_count = machine.get("gpuCount")
  if gpu_count is None:
    gpu_count = 1 if (machine.get("gpuDisplayName") or pod.get("gpuType") or gpu_info.get("id")) else 0

  gpu_name = (
      machine.get("gpuDisplayName")
      or pod.get("gpuType")
      or gpu_info.get("id")
  )
  if gpu_count == 0:
    gpu_name = None

  cost = pod.get("costPerHr") if pod.get("costPerHr") is not None else pod.get("cost")
  vcpu = pod.get("vcpuCount") or cpu_info.get("vcpuCount") or gpu_info.get("vcpuCount")
  mem = pod.get("memoryInGb") or cpu_info.get("memory") or gpu_info.get("memory")
  disk = pod.get("containerDiskInGb") or pod.get("volumeInGb") or pod.get("disk")
  image = pod.get("imageName") or pod.get("image")

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
      image_name=image,
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
    instances = []
    for p in pods:
      inst = normalize_runpod_instance(p)
      if inst.status == "RUNNING" and ((not inst.ssh_host or not inst.ssh_port) or inst.ssh_host == "ssh.runpod.io") and inst.id:
        try:
          detailed = runpod.get_pod(inst.id)
          if detailed:
            d_inst = normalize_runpod_instance(detailed)
            if d_inst.ssh_host and d_inst.ssh_port:
              inst = d_inst
        except Exception:
          pass
      instances.append(inst)
    return instances

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
