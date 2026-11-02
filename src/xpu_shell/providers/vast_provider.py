import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from xpu_shell.providers.base import (
    BaseProvider,
    Instance,
    Template,
    format_template_identifier,
    parse_template_identifier
)


VAST_API_BASE = "https://console.vast.ai/api/v0"
VAST_API_V1_BASE = "https://console.vast.ai/api/v1"


def find_vast_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
  if explicit_key:
    return explicit_key.strip()

  env_key = os.environ.get("VAST_API_KEY")
  if env_key:
    return env_key.strip()

  vast_key_file = Path.home() / ".vast_api_key"
  if vast_key_file.is_file():
    try:
      return vast_key_file.read_text().strip()
    except Exception:
      pass

  return None


def normalize_vast_status(raw_status: Optional[str]) -> str:
  if not raw_status:
    return "UNKNOWN"
  st = raw_status.lower()
  if st in ("running", "active"):
    return "RUNNING"
  if st in ("stopped", "exited"):
    return "STOPPED"
  if st in ("loading", "creating", "starting", "init"):
    return "INITIALIZING"
  if st in ("offline", "inactive"):
    return "OFFLINE"
  if st in ("deleted", "destroying", "terminated"):
    return "TERMINATED"
  return raw_status.upper()


def normalize_vast_instance(data: Dict[str, Any]) -> Instance:
  instance_id = str(data.get("id"))
  actual_status = data.get("actual_status") or data.get("cur_state") or data.get("status")
  status = normalize_vast_status(actual_status)

  ssh_host = data.get("ssh_host")
  ssh_port = data.get("ssh_port")
  if ssh_port is not None:
    try:
      ssh_port = int(ssh_port)
    except (ValueError, TypeError):
      pass

  gpu_name = data.get("gpu_name")
  gpu_count = data.get("num_gpus") or 1
  cost = data.get("dph_total")
  vcpu = data.get("cpu_cores") or data.get("cpu_cores_effective")
  mem_mb = data.get("cpu_ram")
  mem_gb = float(mem_mb) / 1024.0 if mem_mb is not None else None
  disk = data.get("disk_space")
  image = data.get("image") or data.get("image_uuid") or data.get("image_name")
  label = data.get("label") or f"vast-{instance_id}"

  duration = data.get("duration")
  uptime_sec = None
  if duration is not None:
    try:
      uptime_sec = int(float(duration))
    except (ValueError, TypeError):
      pass
  elif data.get("uptime_mins"):
    try:
      uptime_sec = int(float(data.get("uptime_mins")) * 60)
    except (ValueError, TypeError):
      pass

  return Instance(
      id=instance_id,
      name=label,
      provider="vast",
      status=status,
      ssh_host=ssh_host,
      ssh_port=ssh_port,
      gpu_name=gpu_name,
      gpu_count=int(gpu_count),
      uptime_seconds=uptime_sec,
      cost_per_hr=float(cost) if cost is not None else None,
      vcpu_count=int(vcpu) if vcpu is not None else None,
      memory_gb=mem_gb,
      disk_gb=float(disk) if disk is not None else None,
      image_name=image,
      raw=data
  )


class VastProvider(BaseProvider):

  def __init__(self, api_key: Optional[str] = None):
    self._api_key = find_vast_api_key(api_key)
    self._session = requests.Session()
    self._session.headers.update({
        "Accept": "application/json"
    })

  @property
  def name(self) -> str:
    return "vast"

  @property
  def default_image(self) -> Optional[str]:
    return "pytorch/pytorch:latest"

  def register_create_args(self, parser: Any) -> None:
    parser.add_argument(
        "--offer-id",
        type=int,
        help="Direct Vast.ai offer/ask ID to rent (optional for Vast.ai)."
    )

  def _get_headers(self) -> Dict[str, str]:
    if not self._api_key:
      raise ValueError(
          "Vast.ai API key not found. Please provide --api-key, set VAST_API_KEY environment variable, "
          "or place your key in ~/.vast_api_key."
      )
    return {"Authorization": f"Bearer {self._api_key}"}

  def _request(
      self,
      method: str,
      endpoint: str,
      api_version: str = "v0",
      **kwargs
  ) -> Any:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
      url = endpoint
    elif endpoint.startswith("/api/"):
      url = f"https://console.vast.ai{endpoint}"
    else:
      base = VAST_API_V1_BASE if api_version == "v1" else VAST_API_BASE
      url = f"{base}{endpoint}"
    headers = self._get_headers()
    if "headers" in kwargs:
      headers.update(kwargs.pop("headers"))

    resp = self._session.request(method, url, headers=headers, **kwargs)
    if resp.status_code >= 400:
      error_msg = resp.text
      try:
        err_json = resp.json()
        error_msg = err_json.get("msg") or err_json.get("error") or error_msg
      except Exception:
        pass
      raise RuntimeError(f"Vast.ai API error ({resp.status_code}): {error_msg}")

    if not resp.content:
      return {}
    return resp.json()

  def list_instances(self) -> List[Instance]:
    data = self._request("GET", "/instances/", api_version="v1")
    instances = data.get("instances", []) if isinstance(data, dict) else data
    if not isinstance(instances, list):
      instances = []
    return [normalize_vast_instance(item) for item in instances]

  def get_instance(self, instance_id: str) -> Instance:
    clean_id = str(instance_id).strip()
    try:
      data = self._request("GET", f"/instances/{clean_id}/")
      if isinstance(data, dict) and data.get("id"):
        return normalize_vast_instance(data)
      if isinstance(data, dict) and "instances" in data:
        for inst in data["instances"]:
          if str(inst.get("id")) == clean_id:
            return normalize_vast_instance(inst)
    except Exception:
      pass

    all_instances = self.list_instances()
    for inst in all_instances:
      if inst.id == clean_id:
        return inst

    raise FileNotFoundError(f"Vast.ai instance '{instance_id}' not found.")

  def get_ssh_endpoint(self, instance_id: str) -> Tuple[Optional[str], Optional[int]]:
    instance = self.get_instance(instance_id)
    return instance.ssh_host, instance.ssh_port

  def search_offers(
      self,
      gpu_name: Optional[str] = None,
      min_gpus: int = 1,
      min_disk_gb: Optional[float] = None,
      min_reliability: float = 0.90,
      verified_only: bool = False
  ) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {
        "rentable": {"eq": True},
        "rented": {"eq": False}
    }
    params = {"q": json.dumps(query)}
    data = self._request("GET", "/bundles/", params=params)
    offers = data.get("offers", []) if isinstance(data, dict) else data
    if not isinstance(offers, list):
      offers = []

    filtered = []
    for o in offers:
      if o.get("rented"):
        continue
      if verified_only and not o.get("verified"):
        continue
      if o.get("reliability2", 1.0) < min_reliability:
        continue
      if o.get("num_gpus", 1) < min_gpus:
        continue
      if min_disk_gb and (o.get("disk_space") or 0) < min_disk_gb:
        continue
      if gpu_name:
        off_gpu = str(o.get("gpu_name", ""))
        pattern = gpu_name.replace(" ", "").lower()
        if pattern not in off_gpu.replace(" ", "").lower():
          continue
      filtered.append(o)

    filtered.sort(key=lambda x: x.get("dph_total", float("inf")))
    return filtered

  def create_instance(
      self,
      gpu_name: Optional[str] = None,
      image_name: Optional[str] = None,
      template: Optional[str] = None,
      template_id: Optional[str] = None,
      disk_gb: float = 50.0,
      env: Optional[Dict[str, str]] = None,
      offer_id: Optional[int] = None,
      gpu_count: int = 1,
      min_reliability: float = 0.90,
      **kwargs
  ) -> Instance:
    if template:
      kind, val = parse_template_identifier(template)
      if kind == "image":
        image_name = val
      elif kind == "template":
        template_id = val

    target_offer_id = offer_id
    if target_offer_id is None:
      offers = self.search_offers(
          gpu_name=gpu_name,
          min_gpus=gpu_count,
          min_disk_gb=disk_gb,
          min_reliability=min_reliability
      )
      if not offers:
        raise RuntimeError(
            f"No matching Vast.ai offers found for GPU: '{gpu_name or 'any'}' (min {gpu_count} GPUs, {disk_gb} GB disk)."
        )
      target_offer_id = offers[0]["id"]

    body: Dict[str, Any] = {
        "client_id": "me",
        "disk": disk_gb,
        "runtype": "ssh"
    }
    if template_id:
      if str(template_id).isdigit():
        body["template_id"] = int(template_id)
      else:
        body["template_hash_id"] = str(template_id)

    resolved_image = image_name or (None if template_id else self.default_image)
    if resolved_image:
      body["image"] = resolved_image

    if env:
      body["env"] = env

    res = self._request("PUT", f"/asks/{target_offer_id}/", json=body)
    new_contract = res.get("new_contract") or res.get("id")
    if not new_contract:
      raise RuntimeError(f"Vast.ai did not return an instance ID after renting offer {target_offer_id}: {res}")

    return self.get_instance(str(new_contract))

  def stop_instance(self, instance_id: str) -> None:
    self._request("PUT", f"/instances/{instance_id}/", json={"state": "stopped"})

  def start_instance(self, instance_id: str) -> None:
    self._request("PUT", f"/instances/{instance_id}/", json={"state": "running"})

  def terminate_instance(self, instance_id: str) -> None:
    self._request("DELETE", f"/instances/{instance_id}/")

  def list_gpus(self, filter_pattern: Optional[str] = None, expand: bool = False) -> List[Dict[str, Any]]:
    offers = self.search_offers(min_reliability=0.5)
    if expand:
      if filter_pattern:
        regex = re.compile(filter_pattern, re.IGNORECASE)
        offers = [
            o for o in offers
            if regex.search(str(o.get("gpu_name") or "")) or regex.search(str(o.get("id") or ""))
        ]
      return offers

    gpu_aggregates: Dict[str, Dict[str, Any]] = {}

    for o in offers:
      gname = o.get("gpu_name") or "Unknown"
      dph = o.get("dph_total") or 0.0
      vram = o.get("gpu_ram") or 0

      if gname not in gpu_aggregates:
        gpu_aggregates[gname] = {
            "id": gname,
            "displayName": gname,
            "memoryInGb": vram // 1024 if vram > 128 else vram,
            "minPrice": dph,
            "maxPrice": dph,
            "count": 1
        }
      else:
        agg = gpu_aggregates[gname]
        agg["minPrice"] = min(agg["minPrice"], dph)
        agg["maxPrice"] = max(agg["maxPrice"], dph)
        agg["count"] += 1

    result = list(gpu_aggregates.values())
    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      result = [g for g in result if regex.search(g["displayName"])]
    return result

  def list_templates(self, filter_pattern: Optional[str] = None) -> List[Template]:
    templates: List[Template] = []
    try:
      res = self._request("GET", "/template/", params={"q": "{}"})
      raw_tpls = res.get("templates", []) if isinstance(res, dict) else []
      for t in raw_tpls:
        t_id = str(t.get("hash_id") or t.get("id") or "")
        name = t.get("name") or t_id
        img = t.get("image")
        desc = t.get("desc")
        templates.append(Template(
            id=format_template_identifier("template", t_id),
            name=name,
            provider="vast",
            type="template",
            image_name=img,
            description=desc,
            raw=t
        ))
    except Exception:
      pass

    common_images = [
        ("pytorch/pytorch:latest", "PyTorch Latest"),
        ("nvidia/cuda:12.2.0-devel-ubuntu22.04", "CUDA 12.2 Devel Ubuntu 22.04"),
        ("tensorflow/tensorflow:latest-gpu", "TensorFlow Latest GPU"),
    ]
    for img_name, desc in common_images:
      templates.append(Template(
          id=format_template_identifier("image", img_name),
          name=desc,
          provider="vast",
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
