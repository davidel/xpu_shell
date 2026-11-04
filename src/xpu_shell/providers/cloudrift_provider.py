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


CLOUDRIFT_API_BASE = "https://api.cloudrift.ai"
CLOUDRIFT_API_VERSION = "~upcoming"


def find_cloudrift_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
  if explicit_key:
    return explicit_key.strip()

  env_key = os.environ.get("CLOUDRIFT_API_KEY")
  if env_key:
    return env_key.strip()

  key_file = Path.home() / ".cloudrift_api_key"
  if key_file.is_file():
    try:
      return key_file.read_text().strip()
    except Exception:
      pass

  return None


def normalize_cloudrift_status(raw_status: Optional[str]) -> str:
  if not raw_status:
    return "UNKNOWN"
  st = raw_status.lower()
  if st in ("active", "running"):
    return "RUNNING"
  if st in ("initializing", "creating", "deploying", "pending"):
    return "INITIALIZING"
  if st in ("deactivating", "inactive", "stopped", "terminated", "deleted"):
    return "TERMINATED"
  if st in ("failed", "error", "dead"):
    return "FAILED"
  return raw_status.upper()


def extract_cloudrift_ssh_endpoint(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
  ssh_host = data.get("host_address") or data.get("internal_host_address") or data.get("ip")
  ssh_port = None

  port_mappings = data.get("port_mappings")
  if isinstance(port_mappings, list):
    for mapping in port_mappings:
      if isinstance(mapping, (list, tuple)) and len(mapping) >= 2:
        try:
          h_port, c_port = int(mapping[0]), int(mapping[1])
          if c_port == 22:
            ssh_port = h_port
            break
        except (ValueError, TypeError):
          pass

  if ssh_port is None:
    ssh_port = data.get("ssh_port") or data.get("port")

  if ssh_host and ssh_port is None:
    ssh_port = 22

  if ssh_port is not None:
    try:
      ssh_port = int(ssh_port)
    except (ValueError, TypeError):
      ssh_port = None

  return ssh_host, ssh_port


def normalize_cloudrift_instance(data: Dict[str, Any]) -> Instance:
  instance_id = str(data.get("id") or "")
  name = data.get("instance_name") or data.get("name") or f"cloudrift-{instance_id}"
  raw_status = data.get("status")
  status = normalize_cloudrift_status(raw_status)

  ssh_host, ssh_port = extract_cloudrift_ssh_endpoint(data)

  gpus = data.get("gpus") or []
  gpu_name = None
  gpu_count = data.get("gpu_limit") or 1
  if isinstance(gpus, list) and gpus:
    gpu_count = len(gpus)
    first_gpu = gpus[0]
    if isinstance(first_gpu, dict):
      gpu_name = first_gpu.get("brand_short") or first_gpu.get("brand") or first_gpu.get("vendor")

  if not gpu_name:
    res_info = data.get("resource_info") or {}
    gpu_name = res_info.get("instance_type") or data.get("instance_type")

  cost = data.get("cost_per_hour")
  if cost is None:
    res_info = data.get("resource_info") or {}
    cost = res_info.get("cost_per_hour")
  if cost is not None:
    try:
      cost_val = float(cost)
      if cost_val >= 1.0:
        cost = cost_val / 100.0
      else:
        cost = cost_val
    except (ValueError, TypeError):
      cost = None

  vcpu = data.get("cpu_limit")
  dram = data.get("dram_limit") or data.get("dram")
  mem_gb = None
  if dram is not None:
    try:
      dram_val = float(dram)
      if dram_val > 1024 * 1024 * 1024:
        mem_gb = dram_val / (1024.0 ** 3)
      elif dram_val > 1024:
        mem_gb = dram_val / 1024.0
      else:
        mem_gb = dram_val
    except (ValueError, TypeError):
      mem_gb = None

  disk = data.get("disk_limit")
  disk_gb = None
  if disk is not None:
    try:
      disk_val = float(disk)
      if disk_val > 1024 * 1024 * 1024:
        disk_gb = disk_val / (1024.0 ** 3)
      else:
        disk_gb = disk_val
    except (ValueError, TypeError):
      disk_gb = None

  image = None
  vms = data.get("virtual_machines")
  if isinstance(vms, list) and vms:
    image = vms[0].get("name")
  if not image:
    containers = data.get("containers")
    if isinstance(containers, list) and containers:
      image = containers[0].get("name")

  return Instance(
      id=instance_id,
      name=name,
      provider="cloudrift",
      status=status,
      ssh_host=ssh_host,
      ssh_port=ssh_port,
      gpu_name=gpu_name,
      gpu_count=int(gpu_count) if gpu_count else 1,
      cost_per_hr=cost,
      vcpu_count=int(vcpu) if vcpu is not None else None,
      memory_gb=mem_gb,
      disk_gb=disk_gb,
      image_name=image,
      raw=data
  )


class CloudRiftProvider(BaseProvider):

  def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
    self._api_key = find_cloudrift_api_key(api_key)
    self._base_url = (base_url or os.environ.get("CLOUDRIFT_API_BASE") or CLOUDRIFT_API_BASE).rstrip("/")
    self._session = requests.Session()
    self._session.headers.update({
        "Accept": "application/json"
    })

  @property
  def name(self) -> str:
    return "cloudrift"

  @property
  def default_image(self) -> Optional[str]:
    return "ubuntu"

  def register_create_args(self, parser: Any) -> None:
    parser.add_argument(
        "--datacenter",
        dest="datacenter",
        default=None,
        help="Target CloudRift datacenter (optional)."
    )

  def _get_headers(self) -> Dict[str, str]:
    if not self._api_key:
      raise ValueError(
          "CloudRift API key not found. Please provide --api-key, set CLOUDRIFT_API_KEY environment variable, "
          "or place your key in ~/.cloudrift_api_key."
      )
    return {
        "X-API-KEY": self._api_key,
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

  def _request(self, method: str, path: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    url = f"{self._base_url}{path}"
    headers = self._get_headers()
    if "headers" in kwargs:
      headers.update(kwargs.pop("headers"))

    payload = {
        "version": CLOUDRIFT_API_VERSION,
        "data": data if data is not None else {}
    }

    resp = self._session.request(method, url, headers=headers, json=payload, **kwargs)
    if resp.status_code >= 400:
      error_msg = resp.text
      try:
        err_json = resp.json()
        error_msg = err_json.get("message") or err_json.get("error") or err_json.get("msg") or error_msg
      except Exception:
        pass
      raise RuntimeError(f"CloudRift API error ({resp.status_code}): {error_msg}")

    if not resp.content:
      return {}

    try:
      resp_json = resp.json()
      if isinstance(resp_json, dict) and "data" in resp_json:
        return resp_json["data"]
      return resp_json
    except Exception:
      return resp.text

  def list_instances(self) -> List[Instance]:
    payload = {
        "selector": {
            "ByStatus": {
                "statuses": ["Active", "Initializing", "Deactivating"]
            }
        },
        "mask": {
            "with_connection_info": True
        }
    }
    data = self._request("POST", "/api/v1/instances/list", data=payload)
    instances = []
    if isinstance(data, dict):
      instances = data.get("instances") or []
    elif isinstance(data, list):
      instances = data

    return [normalize_cloudrift_instance(item) for item in instances if isinstance(item, dict)]

  def get_instance(self, instance_id: str) -> Instance:
    clean_id = str(instance_id).strip()
    payload = {
        "selector": {
            "ById": [clean_id]
        },
        "mask": {
            "with_connection_info": True
        }
    }
    try:
      data = self._request("POST", "/api/v1/instances/list", data=payload)
      instances = []
      if isinstance(data, dict):
        instances = data.get("instances") or []
      elif isinstance(data, list):
        instances = data
      for item in instances:
        if isinstance(item, dict) and str(item.get("id")) == clean_id:
          return normalize_cloudrift_instance(item)
    except Exception:
      pass

    for inst in self.list_instances():
      if inst.id == clean_id:
        return inst

    raise FileNotFoundError(f"CloudRift instance '{instance_id}' not found.")

  def get_ssh_endpoint(self, instance_id: str) -> Tuple[Optional[str], Optional[int]]:
    instance = self.get_instance(instance_id)
    return instance.ssh_host, instance.ssh_port

  def search_offers(
      self,
      gpu_name: Optional[str] = None,
      min_gpus: int = 1,
      datacenter: Optional[str] = None
  ) -> List[Dict[str, Any]]:
    payload = {"selector": "All"}
    try:
      data = self._request("POST", "/api/v1/instance-types/list", data=payload)
    except Exception:
      data = {}

    instance_types = []
    if isinstance(data, dict):
      instance_types = data.get("instance_types") or []
    elif isinstance(data, list):
      instance_types = data

    offers: List[Dict[str, Any]] = []
    for it in instance_types:
      if not isinstance(it, dict):
        continue
      it_name = str(it.get("name") or "")
      brand = str(it.get("brand_short") or it.get("manufacturer") or it_name)
      datacenters = it.get("datacenters") or []
      dc_names = [d.get("name") for d in datacenters if isinstance(d, dict) and d.get("name")]

      if datacenter and dc_names and datacenter not in dc_names:
        continue

      variants = it.get("variants") or []
      if not variants:
        variants = [{
            "cost_per_hour": it.get("cost_per_hour", 0),
            "cpu_count": 0,
            "dram": 0,
            "disk": 0,
            "gpu_count": 1,
            "available_nodes": 1
        }]

      for idx, v in enumerate(variants):
        if not isinstance(v, dict):
          continue
        g_count = v.get("gpu_count")
        if g_count is None:
          g_count = 1
        try:
          g_count = int(g_count)
        except (ValueError, TypeError):
          g_count = 1

        if g_count < min_gpus:
          continue

        cost = v.get("cost_per_hour") or it.get("cost_per_hour") or 0.0
        try:
          cost = float(cost)
          if cost >= 1.0:
            cost = cost / 100.0
        except (ValueError, TypeError):
          cost = 0.0

        dram = v.get("dram") or 0
        try:
          dram_gb = float(dram) / (1024.0 ** 3) if dram > 1024 * 1024 * 1024 else float(dram)
        except (ValueError, TypeError):
          dram_gb = 0.0

        disk = v.get("disk") or 0
        try:
          disk_gb = float(disk) / (1024.0 ** 3) if disk > 1024 * 1024 * 1024 else float(disk)
        except (ValueError, TypeError):
          disk_gb = 0.0

        avail = v.get("available_nodes") or 0

        offer_entry = {
            "id": f"{it_name}-{idx}",
            "offerId": f"{it_name}-{idx}",
            "instanceType": it_name,
            "gpuType": brand,
            "gpuName": brand,
            "gpuModel": brand,
            "displayName": brand,
            "gpuCount": g_count,
            "num_gpus": g_count,
            "pricePerHour": cost,
            "price": cost,
            "dph_total": cost,
            "memoryInGb": int(dram_gb),
            "gpu_ram": int(dram_gb * 1024),
            "disk": disk_gb,
            "disk_space": disk_gb,
            "location": ", ".join(dc_names) if dc_names else "Any",
            "region": dc_names[0] if dc_names else "default",
            "country": dc_names[0] if dc_names else "default",
            "reliability": f"{avail} available",
            "available_nodes": avail,
            "raw": it
        }
        offers.append(offer_entry)

    if gpu_name:
      req_name = gpu_name.lower().replace("-", "").replace(" ", "").replace("_", "")
      filtered = []
      for o in offers:
        cand_type = str(o.get("instanceType", "")).lower().replace("-", "").replace(" ", "").replace("_", "")
        cand_brand = str(o.get("gpuType", "")).lower().replace("-", "").replace(" ", "").replace("_", "")
        if req_name in cand_type or req_name in cand_brand or cand_type in req_name or cand_brand in req_name:
          filtered.append(o)
      if filtered:
        offers = filtered

    def get_price(item: Dict[str, Any]) -> float:
      price = item.get("pricePerHour")
      try:
        return float(price)
      except (ValueError, TypeError):
        return float("inf")

    offers.sort(key=get_price)
    return offers

  def create_instance(
      self,
      name: str = "persistent-worker",
      gpu_name: Optional[str] = None,
      image_name: Optional[str] = None,
      template: Optional[str] = None,
      disk_gb: float = 50.0,
      env: Optional[Dict[str, str]] = None,
      ssh_public_key: Optional[str] = None,
      datacenter: Optional[str] = None,
      gpu_count: int = 1,
      **kwargs
  ) -> Instance:
    if template:
      kind, val = parse_template_identifier(template)
      image_name = val

    offers = self.search_offers(
        gpu_name=gpu_name,
        min_gpus=gpu_count,
        datacenter=datacenter
    )

    if offers:
      target_instance_type = offers[0].get("instanceType") or gpu_name or "rtx4090"
    else:
      target_instance_type = gpu_name or "rtx4090"

    is_docker = False
    if image_name:
      if "/" in image_name or ":" in image_name:
        is_docker = True

    if is_docker:
      config: Dict[str, Any] = {
          "Docker": {
              "image": image_name
          }
      }
    else:
      vm_config: Dict[str, Any] = {
          "image_url": image_name or self.default_image,
      }
      if ssh_public_key:
        vm_config["ssh_key"] = {
            "PublicKeys": [ssh_public_key]
        }
      config = {
          "VirtualMachine": vm_config
      }

    selector: Dict[str, Any] = {
        "ByInstanceTypeAndLocation": {
            "instance_type": target_instance_type
        }
    }
    if datacenter:
      selector["ByInstanceTypeAndLocation"]["datacenters"] = [datacenter]

    rent_payload: Dict[str, Any] = {
        "selector": selector,
        "with_public_ip": True,
        "config": config,
        "name": name
    }

    res = self._request("POST", "/api/v1/instances/rent", data=rent_payload)
    new_ids = []
    if isinstance(res, dict):
      new_ids = res.get("instance_ids") or []
      if not new_ids and res.get("id"):
        new_ids = [res.get("id")]
    elif isinstance(res, list):
      new_ids = res

    if not new_ids:
      raise RuntimeError(f"CloudRift did not return any instance IDs after rent request: {res}")

    new_id = str(new_ids[0])
    return self.get_instance(new_id)

  def stop_instance(self, instance_id: str) -> None:
    self.terminate_instance(instance_id)

  def start_instance(self, instance_id: str) -> None:
    raise NotImplementedError(
        f"CloudRift does not support starting terminated instance '{instance_id}'. Please create a new instance."
    )

  def terminate_instance(self, instance_id: str) -> None:
    clean_id = str(instance_id).strip()
    payload = {
        "selector": {
            "ById": [clean_id]
        }
    }
    self._request("POST", "/api/v1/instances/terminate", data=payload)

  def list_gpus(self, filter_pattern: Optional[str] = None, expand: bool = False) -> List[Dict[str, Any]]:
    offers = self.search_offers()
    if expand:
      if filter_pattern:
        regex = re.compile(filter_pattern, re.IGNORECASE)
        offers = [
            o for o in offers
            if regex.search(str(o.get("gpuType") or o.get("gpuName") or ""))
            or regex.search(str(o.get("displayName") or ""))
            or regex.search(str(o.get("id") or o.get("offerId") or ""))
            or regex.search(str(o.get("instanceType") or ""))
        ]
      return offers

    gpu_aggregates: Dict[str, Dict[str, Any]] = {}
    for o in offers:
      gname = str(o.get("gpuType") or o.get("gpuName") or "Unknown")
      price = float(o.get("pricePerHour") or 0.0)
      vram = int(o.get("memoryInGb") or 0)
      nodes = int(o.get("available_nodes") or 0)

      if gname not in gpu_aggregates:
        gpu_aggregates[gname] = {
            "id": str(o.get("instanceType") or gname),
            "displayName": gname,
            "memoryInGb": vram,
            "minPrice": price,
            "maxPrice": price,
            "count": nodes if nodes > 0 else 1
        }
      else:
        agg = gpu_aggregates[gname]
        agg["minPrice"] = min(agg["minPrice"], price)
        agg["maxPrice"] = max(agg["maxPrice"], price)
        agg["count"] += nodes if nodes > 0 else 1

    result = list(gpu_aggregates.values())
    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      result = [
          g for g in result
          if regex.search(g.get("displayName", "")) or regex.search(g.get("id", ""))
      ]
    return result

  def list_templates(self, filter_pattern: Optional[str] = None) -> List[Template]:
    templates: List[Template] = []
    try:
      data = self._request("POST", "/api/v1/recipes/list", data={})
      groups = []
      if isinstance(data, dict):
        groups = data.get("groups") or []
      elif isinstance(data, list):
        groups = data

      for g in groups:
        if not isinstance(g, dict):
          continue
        recipes = g.get("recipes") or []
        for r in recipes:
          if not isinstance(r, dict):
            continue
          r_name = str(r.get("name") or "")
          if not r_name:
            continue
          templates.append(
              Template(
                  id=format_template_identifier("template", r_name),
                  name=r_name,
                  provider="cloudrift",
                  type="template",
                  image_name=r_name,
                  raw=r
              )
          )
    except Exception:
      pass

    if not templates:
      default_os = [
          ("ubuntu", "Ubuntu Linux (Default)"),
          ("debian", "Debian Linux"),
      ]
      templates = [
          Template(
              id=format_template_identifier("image", os_name),
              name=desc,
              provider="cloudrift",
              type="image",
              image_name=os_name
          )
          for os_name, desc in default_os
      ]

    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      templates = [
          t for t in templates
          if regex.search(t.id) or regex.search(t.name) or regex.search(t.image_name or "")
      ]
    return templates
