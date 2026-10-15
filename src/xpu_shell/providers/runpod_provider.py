import os
import re
from typing import Any, Dict, List, Optional, Tuple
import runpod

from xpu_shell.providers.base import BaseProvider, Instance


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

  def list_gpus(self, filter_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
    gpus = runpod.get_gpus() or []
    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      gpus = [g for g in gpus if regex.search(g.get("id", "")) or regex.search(g.get("displayName", ""))]
    return gpus
