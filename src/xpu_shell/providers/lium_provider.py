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


LIUM_API_BASE = "https://lium.io/api"


def find_lium_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
  if explicit_key:
    return explicit_key.strip()

  env_key = os.environ.get("LIUM_API_KEY")
  if env_key:
    return env_key.strip()

  for p in [
      Path.home() / ".lium_api_key",
      Path.home() / ".config" / "lium" / "api_key",
      Path.home() / ".lium" / "token"
  ]:
    if p.is_file():
      try:
        content = p.read_text().strip()
        if content:
          return content
      except Exception:
        pass

  return None


def normalize_lium_status(raw_status: Optional[str]) -> str:
  if not raw_status:
    return "UNKNOWN"
  st = raw_status.upper().strip()
  if st in ("RUNNING", "ACTIVE"):
    return "RUNNING"
  if st in ("PENDING", "START_PENDING", "INITIALIZING"):
    return "INITIALIZING"
  if st in ("STOPPED", "STOP_PENDING"):
    return "STOPPED"
  if st in ("DELETING", "DELETED", "TERMINATED"):
    return "TERMINATED"
  if st in ("FAILED", "CREATION_FAILED", "BROKEN", "REBOOT_FAILED"):
    return "FAILED"
  return st


def extract_lium_ssh_endpoint(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
  cmd = data.get("ssh_connect_cmd")
  host = None
  port = None

  if cmd and isinstance(cmd, str):
    port_m = re.search(r'-p\s*(\d+)', cmd)
    if port_m:
      try:
        port = int(port_m.group(1))
      except (ValueError, TypeError):
        port = None

    cleaned = re.sub(r'-[a-zA-Z]\s+[^\s]+', '', cmd)
    tokens = [t for t in cleaned.split() if t != "ssh" and not t.startswith("-")]
    for t in tokens:
      if "@" in t:
        host = t.split("@", 1)[1].strip()
        break
      elif "." in t or ":" in t:
        host = t.strip()
        break
    if not host and tokens:
      host = tokens[-1].strip()

  if port is None:
    ports_map = data.get("ports_mapping") or {}
    if isinstance(ports_map, dict):
      for k, v in ports_map.items():
        if str(k).split("/")[0] == "22":
          try:
            port = int(v)
            break
          except (ValueError, TypeError):
            pass

  if not host:
    host = data.get("executor_ip_address") or data.get("executor_checked_ip") or data.get("ip")

  if host and port is None:
    port = 22

  return host, port


def normalize_lium_instance(data: Dict[str, Any]) -> Instance:
  instance_id = str(data.get("id") or data.get("pod_id") or "")
  name = data.get("pod_name") or data.get("name") or f"lium-{instance_id}"
  raw_status = data.get("status")
  status = normalize_lium_status(raw_status)

  ssh_host, ssh_port = extract_lium_ssh_endpoint(data)

  gpu_name = data.get("gpu_name")
  raw_count = data.get("gpu_count")
  gpu_count = 1
  if raw_count is not None:
    try:
      gpu_count = int(raw_count)
    except (ValueError, TypeError):
      gpu_count = 1

  cost = data.get("price") or data.get("price_per_hour")
  cost_per_hr = None
  if cost is not None:
    try:
      cost_per_hr = float(cost)
    except (ValueError, TypeError):
      cost_per_hr = None

  vcpu = data.get("cpu_count")
  vcpu_count = None
  if vcpu is not None:
    try:
      vcpu_count = int(vcpu)
    except (ValueError, TypeError):
      vcpu_count = None

  ram = data.get("ram_total")
  memory_gb = None
  if ram is not None:
    try:
      ram_val = float(ram)
      if ram_val > 1024 * 1024 * 1024:
        memory_gb = ram_val / (1024.0 ** 3)
      elif ram_val > 1024:
        memory_gb = ram_val / 1024.0
      else:
        memory_gb = ram_val
    except (ValueError, TypeError):
      memory_gb = None

  disk = data.get("storage_limit_gb") or data.get("volume_limit_gb")
  disk_gb = None
  if disk is not None:
    try:
      disk_gb = float(disk)
    except (ValueError, TypeError):
      disk_gb = None

  image = data.get("docker_image") or data.get("template_name") or data.get("template_id")

  return Instance(
      id=instance_id,
      name=name,
      provider="lium",
      status=status,
      ssh_host=ssh_host,
      ssh_port=ssh_port,
      gpu_name=gpu_name,
      gpu_count=gpu_count,
      cost_per_hr=cost_per_hr,
      vcpu_count=vcpu_count,
      memory_gb=memory_gb,
      disk_gb=disk_gb,
      image_name=image,
      raw=data
  )


class LiumProvider(BaseProvider):

  def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
    self._api_key = find_lium_api_key(api_key)
    self._base_url = (base_url or os.environ.get("LIUM_API_BASE") or LIUM_API_BASE).rstrip("/")
    self._session = requests.Session()
    self._session.headers.update({
        "Accept": "application/json"
    })

  @property
  def name(self) -> str:
    return "lium"

  @property
  def default_image(self) -> Optional[str]:
    return "daturaai/pytorch:latest"

  def register_create_args(self, parser: Any) -> None:
    parser.add_argument(
        "--executor-id",
        dest="executor_id",
        default=None,
        help="Target Lium executor UUID (optional)."
    )

  def _get_headers(self, require_auth: bool = True) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    if self._api_key:
      headers["X-API-Key"] = self._api_key
    elif require_auth:
      raise ValueError(
          "Lium API key not found. Please provide --api-key, set LIUM_API_KEY environment variable, "
          "or place your key in ~/.lium_api_key."
      )
    return headers

  def _request(
      self,
      method: str,
      path: str,
      data: Optional[Dict[str, Any]] = None,
      require_auth: bool = True,
      **kwargs
  ) -> Any:
    url = f"{self._base_url}{path}"
    headers = self._get_headers(require_auth=require_auth)
    if "headers" in kwargs:
      headers.update(kwargs.pop("headers"))

    resp = self._session.request(
        method,
        url,
        headers=headers,
        json=data if data is not None else None,
        **kwargs
    )
    if resp.status_code >= 400:
      error_msg = resp.text
      try:
        err_json = resp.json()
        error_msg = err_json.get("detail") or err_json.get("message") or err_json.get("error") or error_msg
      except Exception:
        pass
      raise RuntimeError(f"Lium API error ({resp.status_code}): {error_msg}")

    if not resp.content:
      return {}

    try:
      return resp.json()
    except Exception:
      return resp.text

  def list_instances(self) -> List[Instance]:
    res = self._request("GET", "/pods")
    items = []
    if isinstance(res, list):
      items = res
    elif isinstance(res, dict):
      items = res.get("items") or res.get("pods") or []

    instances = []
    for item in items:
      if isinstance(item, dict):
        instances.append(normalize_lium_instance(item))
    return instances

  def get_instance(self, instance_id: str) -> Instance:
    clean_id = str(instance_id).strip()
    if clean_id.lower().startswith("lium:"):
      clean_id = clean_id.split(":", 1)[1]
    res = self._request("GET", f"/pods/{clean_id}")
    if isinstance(res, dict):
      return normalize_lium_instance(res)
    raise RuntimeError(f"Unexpected response format when fetching pod '{clean_id}': {res}")

  def get_ssh_endpoint(self, instance_id: str) -> Tuple[Optional[str], Optional[int]]:
    inst = self.get_instance(instance_id)
    return inst.ssh_host, inst.ssh_port

  def search_offers(
      self,
      gpu_name: Optional[str] = None,
      min_gpus: int = 1,
      min_reliability: float = 0.0
  ) -> List[Dict[str, Any]]:
    try:
      data = self._request("GET", "/executors", require_auth=False)
    except Exception:
      data = []

    executors = []
    if isinstance(data, list):
      executors = data
    elif isinstance(data, dict):
      executors = data.get("items") or data.get("executors") or []

    offers: List[Dict[str, Any]] = []
    for ex in executors:
      if not isinstance(ex, dict):
        continue

      active = ex.get("active", True)
      if not active:
        continue

      specs = ex.get("specs") or {}
      gpu_info = specs.get("gpu") or {}
      details = gpu_info.get("details") or []
      gpu_model = None
      vram_mb = 0
      if details and isinstance(details, list):
        first_det = details[0]
        if isinstance(first_det, dict):
          gpu_model = first_det.get("name")
          vram_mb = first_det.get("capacity") or 0

      if not gpu_model:
        gpu_model = ex.get("machine_name") or "Unknown GPU"

      vram_gb = int(vram_mb // 1024) if vram_mb > 128 else int(vram_mb)

      raw_g_count = ex.get("gpu_count") or gpu_info.get("count") or 1
      try:
        g_count = int(raw_g_count)
      except (ValueError, TypeError):
        g_count = 1

      if g_count < min_gpus:
        continue

      price_per_gpu = ex.get("price_per_gpu")
      if price_per_gpu is not None:
        try:
          price_total = float(price_per_gpu) * float(g_count)
        except (ValueError, TypeError):
          price_total = 0.0
      else:
        try:
          price_total = float(ex.get("price_per_hour") or 0.0)
        except (ValueError, TypeError):
          price_total = 0.0

      reliability_score = ex.get("reliability_score")
      rel_float = 1.0
      if reliability_score is not None:
        try:
          rel_float = float(reliability_score)
        except (ValueError, TypeError):
          rel_float = 1.0

      if rel_float < min_reliability:
        continue

      net_info = specs.get("network") or {}
      down = net_info.get("ema_verifyx_download_speed") or net_info.get("download_speed")
      up = net_info.get("ema_verifyx_upload_speed") or net_info.get("upload_speed")

      disk_info = specs.get("hard_disk") or {}
      total_disk = disk_info.get("total") or 0
      try:
        disk_gb = float(total_disk) / (1024.0 ** 3) if total_disk > 1024 * 1024 * 1024 else float(total_disk)
      except (ValueError, TypeError):
        disk_gb = 50.0

      location = ex.get("location")
      loc_str = str(location) if location else "Any"

      offer_entry = {
          "id": ex.get("id"),
          "offerId": ex.get("id"),
          "instanceType": gpu_model,
          "gpuType": gpu_model,
          "gpuName": gpu_model,
          "gpuModel": gpu_model,
          "displayName": gpu_model,
          "gpuCount": g_count,
          "num_gpus": g_count,
          "pricePerHour": price_total,
          "price": price_total,
          "dph_total": price_total,
          "memoryInGb": vram_gb,
          "gpu_ram": vram_mb,
          "disk": int(disk_gb) if disk_gb else 50,
          "disk_space": int(disk_gb) if disk_gb else 50,
          "location": loc_str,
          "region": loc_str,
          "country": loc_str,
          "reliability": f"{rel_float * 100:.1f}%",
          "reliability2": rel_float,
          "inet_down": float(down) if down is not None else None,
          "inet_up": float(up) if up is not None else None,
          "available_nodes": 1,
          "raw": ex
      }
      offers.append(offer_entry)

    if gpu_name:
      req_name = gpu_name.lower().replace("-", "").replace(" ", "").replace("_", "")
      filtered = []
      for o in offers:
        cand_brand = str(o.get("gpuType", "")).lower().replace("-", "").replace(" ", "").replace("_", "")
        if req_name in cand_brand or cand_brand in req_name:
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
      executor_id: Optional[str] = None,
      gpu_count: int = 1,
      min_cpus: Optional[int] = None,
      min_ram_gb: Optional[float] = None,
      country: Optional[str] = None,
      termination_hours: Optional[int] = None,
      dockerfile_content: Optional[str] = None,
      **kwargs
  ) -> Instance:
    template_id = None
    if template:
      kind, val = parse_template_identifier(template)
      if val:
        if re.match(r'^[0-9a-fA-F-]{36}$', val):
          template_id = val
        else:
          for t in self.list_templates():
            if t.name.lower() == val.lower() or t.id.lower() == val.lower():
              template_id = t.id
              break
          if not template_id:
            template_id = val

    if not ssh_public_key:
      for key_name in ["id_ed25519.pub", "id_rsa.pub", "id_ecdsa.pub"]:
        pub_path = Path.home() / ".ssh" / key_name
        if pub_path.is_file():
          try:
            ssh_public_key = pub_path.read_text().strip()
            break
          except Exception:
            pass

    if not ssh_public_key:
      raise ValueError("SSH public key is required to rent a Lium pod. None found in ~/.ssh/")

    if executor_id:
      rent_payload: Dict[str, Any] = {
          "pod_name": name,
          "user_public_key": ssh_public_key,
          "gpu_count": int(gpu_count)
      }
      if template_id:
        rent_payload["template_id"] = template_id
      if termination_hours:
        rent_payload["termination_hours"] = int(termination_hours)
      if dockerfile_content:
        rent_payload["dockerfile_content"] = dockerfile_content

      res = self._request("POST", f"/executors/{executor_id}/rent", data=rent_payload)
    else:
      spec_payload: Dict[str, Any] = {
          "pod_name": name,
          "gpu_type": gpu_name or "RTX4090",
          "gpu_count": int(gpu_count),
          "user_public_key": ssh_public_key
      }
      if template_id:
        spec_payload["template_id"] = template_id
      if disk_gb:
        spec_payload["min_disk_gb"] = float(disk_gb)
      if min_ram_gb:
        spec_payload["min_ram_gb"] = float(min_ram_gb)
      if min_cpus:
        spec_payload["min_cpus"] = int(min_cpus)
      if country:
        spec_payload["country"] = country
      if termination_hours:
        spec_payload["termination_hours"] = int(termination_hours)
      if dockerfile_content:
        spec_payload["dockerfile_content"] = dockerfile_content

      res = self._request("POST", "/executors/rent-by-spec", data=spec_payload)

    pod_id = None
    if isinstance(res, dict):
      pod_id = res.get("pod_id") or res.get("id")

    if pod_id:
      try:
        return self.get_instance(str(pod_id))
      except Exception:
        return Instance(
            id=str(pod_id),
            name=name,
            provider="lium",
            status="INITIALIZING",
            gpu_name=gpu_name,
            gpu_count=gpu_count,
            raw=res if isinstance(res, dict) else {}
        )

    if isinstance(res, dict):
      return normalize_lium_instance(res)

    raise RuntimeError(f"Failed to create Lium instance. API response: {res}")

  def stop_instance(self, instance_id: str) -> None:
    self.terminate_instance(instance_id)

  def start_instance(self, instance_id: str) -> None:
    raise NotImplementedError(
        f"Lium pods cannot be resumed once stopped or terminated ('{instance_id}'). Please create a new pod."
    )

  def terminate_instance(self, instance_id: str) -> None:
    clean_id = str(instance_id).strip()
    if clean_id.lower().startswith("lium:"):
      clean_id = clean_id.split(":", 1)[1]
    self._request("DELETE", f"/pods/{clean_id}")

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
            "id": gname,
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
      data = self._request("GET", "/templates", require_auth=False)
      items = data if isinstance(data, list) else []
      for t in items:
        if not isinstance(t, dict):
          continue
        tid = str(t.get("id") or "")
        tname = str(t.get("name") or tid)
        desc = t.get("description")
        docker_img = t.get("docker_image")
        tag = t.get("docker_image_tag")
        img_str = f"{docker_img}:{tag}" if docker_img and tag else docker_img
        templates.append(
            Template(
                id=format_template_identifier("template", tid),
                name=tname,
                provider="lium",
                type="template",
                image_name=img_str,
                description=desc,
                raw=t
            )
        )
    except Exception:
      pass

    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      templates = [
          t for t in templates
          if regex.search(t.name)
          or regex.search(t.id)
          or (t.image_name and regex.search(t.image_name))
          or (t.description and regex.search(t.description))
      ]
    return templates
