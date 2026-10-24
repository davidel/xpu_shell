import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import requests

from xpu_shell.providers.base import BaseProvider, Instance


SPHERON_API_BASE = "https://app.spheron.ai"


def find_spheron_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
  if explicit_key:
    return explicit_key.strip()

  env_key = os.environ.get("SPHERON_API_KEY")
  if env_key:
    return env_key.strip()

  key_file = Path.home() / ".spheron_api_key"
  if key_file.is_file():
    try:
      return key_file.read_text().strip()
    except Exception:
      pass

  return None


def normalize_spheron_status(raw_status: Optional[str]) -> str:
  if not raw_status:
    return "UNKNOWN"
  st = raw_status.lower()
  if st in ("running", "active"):
    return "RUNNING"
  if st in ("stopped", "parked", "paused", "exited"):
    return "STOPPED"
  if st in ("deploying", "creating", "starting", "initializing", "init"):
    return "INITIALIZING"
  if st in ("offline", "inactive"):
    return "OFFLINE"
  if st in ("terminated", "deleted", "destroying"):
    return "TERMINATED"
  if st in ("failed", "error", "dead"):
    return "FAILED"
  return raw_status.upper()


def extract_ssh_endpoint(data: Dict[str, Any]) -> Tuple[Optional[str], Optional[int]]:
  ssh_host = None
  ssh_port = None

  conn = data.get("connectionDetails") or data.get("connection") or {}
  if isinstance(conn, dict):
    ssh_host = conn.get("ip") or conn.get("host") or conn.get("sshHost")
    ssh_port = conn.get("port") or conn.get("sshPort")

  if not ssh_host:
    ssh_host = (
        data.get("sshHost")
        or data.get("ip")
        or data.get("publicIp")
        or data.get("host")
    )

  if ssh_port is None:
    ssh_port = data.get("sshPort") or data.get("port")

  if ssh_host and ssh_port is None:
    ssh_port = 22

  if ssh_port is not None:
    try:
      ssh_port = int(ssh_port)
    except (ValueError, TypeError):
      ssh_port = None

  return ssh_host, ssh_port


def normalize_spheron_instance(data: Dict[str, Any]) -> Instance:
  instance_id = str(data.get("id") or data.get("deploymentId") or "")
  raw_status = data.get("status") or data.get("state")
  status = normalize_spheron_status(raw_status)

  ssh_host, ssh_port = extract_ssh_endpoint(data)

  gpu_name = data.get("gpuType") or data.get("gpuName") or data.get("gpu")
  gpu_count = data.get("gpuCount") or data.get("numGpus") or 1
  cost = data.get("pricePerHour") or data.get("dph") or data.get("costPerHr")
  vcpu = data.get("vcpu") or data.get("cpu") or data.get("cpuCores")
  mem = data.get("memoryInGb") or data.get("ramGb") or data.get("memory")
  if mem is not None:
    try:
      mem_val = float(mem)
      if mem_val > 256:
        mem = mem_val / 1024.0
      else:
        mem = mem_val
    except (ValueError, TypeError):
      mem = None

  disk = data.get("disk") or data.get("diskInGb") or data.get("storage")
  image = data.get("image") or data.get("imageName") or data.get("operatingSystem")
  name = data.get("name") or data.get("label") or f"spheron-{instance_id}"

  return Instance(
      id=instance_id,
      name=name,
      provider="spheron",
      status=status,
      ssh_host=ssh_host,
      ssh_port=ssh_port,
      gpu_name=gpu_name,
      gpu_count=int(gpu_count),
      cost_per_hr=float(cost) if cost is not None else None,
      vcpu_count=int(vcpu) if vcpu is not None else None,
      memory_gb=mem,
      disk_gb=float(disk) if disk is not None else None,
      image_name=image,
      raw=data
  )


class SpheronProvider(BaseProvider):

  def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
    self._api_key = find_spheron_api_key(api_key)
    self._base_url = (base_url or os.environ.get("SPHERON_API_BASE") or SPHERON_API_BASE).rstrip("/")
    self._session = requests.Session()
    self._session.headers.update({
        "Accept": "application/json"
    })

  @property
  def name(self) -> str:
    return "spheron"

  @property
  def default_image(self) -> Optional[str]:
    return "ubuntu-22.04"

  def register_create_args(self, parser: Any) -> None:
    parser.add_argument(
        "--offer-id",
        dest="offer_id",
        help="Direct Spheron GPU offer ID to rent (optional)."
    )
    parser.add_argument(
        "--ssh-key-id",
        dest="ssh_key_id",
        help="Spheron SSH Key ID (optional, auto-injected if omitted)."
    )
    parser.add_argument(
        "--region",
        dest="region",
        default=None,
        help="Target region for Spheron deployment (optional)."
    )

  def _get_headers(self) -> Dict[str, str]:
    if not self._api_key:
      raise ValueError(
          "Spheron API key not found. Please provide --api-key, set SPHERON_API_KEY environment variable, "
          "or place your key in ~/.spheron_api_key."
      )
    return {
        "Authorization": f"Bearer {self._api_key}",
        "Content-Type": "application/json"
    }

  def _request(
      self,
      method: str,
      endpoint: str,
      **kwargs
  ) -> Any:
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
      url = endpoint
    else:
      clean_endpoint = "/" + endpoint.lstrip("/")
      url = f"{self._base_url}{clean_endpoint}"

    headers = self._get_headers()
    if "headers" in kwargs:
      headers.update(kwargs.pop("headers"))

    resp = self._session.request(method, url, headers=headers, **kwargs)
    if resp.status_code >= 400:
      error_msg = resp.text
      try:
        err_json = resp.json()
        error_msg = err_json.get("message") or err_json.get("error") or err_json.get("msg") or error_msg
      except Exception:
        pass
      raise RuntimeError(f"Spheron API error ({resp.status_code}): {error_msg}")

    if not resp.content:
      return {}
    return resp.json()

  def list_instances(self) -> List[Instance]:
    data = self._request("GET", "/api/deployments")
    deployments = []
    if isinstance(data, list):
      deployments = data
    elif isinstance(data, dict):
      deployments = data.get("deployments") or data.get("data") or data.get("instances") or []

    if not isinstance(deployments, list):
      deployments = []

    return [normalize_spheron_instance(item) for item in deployments]

  def get_instance(self, instance_id: str) -> Instance:
    clean_id = str(instance_id).strip()
    try:
      data = self._request("GET", f"/api/deployments/{clean_id}")
      item = data
      if isinstance(data, dict):
        item = data.get("deployment") or data.get("data") or data
      if isinstance(item, dict) and (item.get("id") or item.get("deploymentId")):
        return normalize_spheron_instance(item)
    except Exception:
      pass

    for inst in self.list_instances():
      if inst.id == clean_id:
        return inst

    raise FileNotFoundError(f"Spheron instance '{instance_id}' not found.")

  def get_ssh_endpoint(self, instance_id: str) -> Tuple[Optional[str], Optional[int]]:
    instance = self.get_instance(instance_id)
    return instance.ssh_host, instance.ssh_port

  def search_offers(
      self,
      gpu_name: Optional[str] = None,
      min_gpus: int = 1,
      min_disk_gb: Optional[float] = None,
      region: Optional[str] = None
  ) -> List[Dict[str, Any]]:
    params: Dict[str, Any] = {"limit": 100}
    if region:
      params["region"] = region

    page = 1
    raw_items = []
    while True:
      params["page"] = page
      try:
        data = self._request("GET", "/api/gpu-offers", params=params)
      except Exception:
        data = self._request("GET", "/api/offers", params=params)

      if isinstance(data, list):
        raw_items.extend(data)
        break
      elif isinstance(data, dict):
        if "offers" in data and isinstance(data["offers"], list):
          raw_items.extend(data["offers"])
          break
        elif "gpuOffers" in data and isinstance(data["gpuOffers"], list):
          raw_items.extend(data["gpuOffers"])
          break
        elif "data" in data and isinstance(data["data"], list):
          raw_items.extend(data["data"])
          total_pages = data.get("totalPages", 1)
          if page >= total_pages:
            break
          page += 1
        else:
          break
      else:
        break

    offers = []
    for item in raw_items:
      if not isinstance(item, dict):
        continue
      sub_offers = item.get("offers")
      if isinstance(sub_offers, list) and sub_offers:
        for sub in sub_offers:
          if not isinstance(sub, dict):
            continue
          o = dict(sub)
          if not o.get("gpuType"):
            o["gpuType"] = item.get("gpuType") or item.get("gpuModel")
          if not o.get("displayName"):
            o["displayName"] = item.get("displayName") or o.get("name")
          offers.append(o)
      else:
        offers.append(dict(item))

    filtered = []
    for o in offers:
      if o.get("available") is False or o.get("rented") is True:
        continue
      count = o.get("gpuCount") or o.get("num_gpus") or o.get("numGpus") or 1
      if count < min_gpus:
        continue

      disk = o.get("disk") or o.get("storage") or o.get("diskInGb") or o.get("disk_space")
      if min_disk_gb and disk is not None and disk < min_disk_gb:
        continue

      clusters = o.get("clusters")
      reg = o.get("region") or (clusters[0] if isinstance(clusters, list) and clusters else None)
      if region:
        reg_clean = str(reg or "").lower()
        if region.lower() not in reg_clean:
          continue

      if gpu_name:
        off_gpu = str(o.get("gpuType") or o.get("gpuName") or o.get("name") or "")
        off_disp = str(o.get("displayName") or "")
        pattern = gpu_name.replace(" ", "").replace("_", "").lower()
        candidates = (off_gpu + " " + off_disp).replace(" ", "").replace("_", "").lower()
        if pattern not in candidates:
          continue

      vram = o.get("gpu_memory") or o.get("memoryInGb") or o.get("ramGb") or o.get("vram") or 0
      try:
        vram = float(vram)
        if vram >= 1024:
          vram = vram / 1024.0
      except (ValueError, TypeError):
        vram = 0

      price = (
          o.get("pricePerHour")
          or o.get("price")
          or o.get("spot_price")
          or o.get("dph")
          or o.get("costPerHr")
          or o.get("dph_total")
          or 0.0
      )
      try:
        price = float(price)
      except (ValueError, TypeError):
        price = 0.0

      o["id"] = o.get("id") or o.get("offerId")
      o["offerId"] = o["id"]
      o["gpuCount"] = count
      o["num_gpus"] = count
      o["memoryInGb"] = int(vram)
      o["gpu_ram"] = int(vram * 1024)
      o["pricePerHour"] = price
      o["price"] = price
      o["dph_total"] = price
      o["disk"] = disk
      o["disk_space"] = disk
      o["region"] = reg
      o["country"] = reg
      o["geolocation"] = reg
      filtered.append(o)

    def get_price(item: Dict[str, Any]) -> float:
      price = item.get("pricePerHour")
      try:
        return float(price)
      except (ValueError, TypeError):
        return float("inf")

    filtered.sort(key=get_price)
    return filtered

  def create_instance(
      self,
      name: str = "persistent-worker",
      gpu_name: Optional[str] = None,
      image_name: Optional[str] = None,
      disk_gb: float = 50.0,
      env: Optional[Dict[str, str]] = None,
      offer_id: Optional[str] = None,
      ssh_key_id: Optional[str] = None,
      ssh_public_key: Optional[str] = None,
      region: Optional[str] = None,
      gpu_count: int = 1,
      **kwargs
  ) -> Instance:
    target_offer_id = offer_id
    target_gpu_type = gpu_name

    if not target_offer_id:
      offers = self.search_offers(
          gpu_name=gpu_name,
          min_gpus=gpu_count,
          min_disk_gb=disk_gb,
          region=region
      )
      if not offers:
        raise RuntimeError(
            f"No matching Spheron offers found for GPU: '{gpu_name or 'any'}' (min {gpu_count} GPUs, {disk_gb} GB disk)."
        )
      best = offers[0]
      target_offer_id = str(best.get("id") or best.get("offerId"))
      target_gpu_type = best.get("gpuType") or best.get("gpuName") or gpu_name

    body: Dict[str, Any] = {
        "name": name,
        "offerId": target_offer_id,
        "gpuType": target_gpu_type,
        "gpuCount": gpu_count,
        "disk": disk_gb,
        "operatingSystem": image_name or self.default_image
    }

    if ssh_key_id:
      body["sshKeyId"] = ssh_key_id
    elif ssh_public_key:
      body["sshKey"] = ssh_public_key
      body["sshPublicKey"] = ssh_public_key

    if region:
      body["region"] = region

    if env:
      body["env"] = env

    res = self._request("POST", "/api/deployments", json=body)
    new_id = res.get("id") or res.get("deploymentId")
    if not new_id and isinstance(res.get("deployment"), dict):
      new_id = res["deployment"].get("id") or res["deployment"].get("deploymentId")

    if not new_id:
      raise RuntimeError(f"Spheron did not return a valid deployment ID: {res}")

    return self.get_instance(str(new_id))

  def stop_instance(self, instance_id: str) -> None:
    try:
      self._request("POST", f"/api/deployments/{instance_id}/stop")
    except Exception:
      try:
        self._request("PATCH", f"/api/deployments/{instance_id}", json={"status": "stopped"})
      except Exception as e:
        raise RuntimeError(f"Spheron failed to stop instance '{instance_id}': {e}") from e

  def start_instance(self, instance_id: str) -> None:
    try:
      self._request("POST", f"/api/deployments/{instance_id}/start")
    except Exception:
      try:
        self._request("PATCH", f"/api/deployments/{instance_id}", json={"status": "running"})
      except Exception as e:
        raise RuntimeError(f"Spheron failed to start instance '{instance_id}': {e}") from e

  def terminate_instance(self, instance_id: str) -> None:
    self._request("DELETE", f"/api/deployments/{instance_id}")

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
        ]
      return offers

    gpu_aggregates: Dict[str, Dict[str, Any]] = {}

    for o in offers:
      gname = str(o.get("gpuType") or o.get("gpuName") or "Unknown")
      price = o.get("pricePerHour") or 0.0
      try:
        price = float(price)
      except (ValueError, TypeError):
        price = 0.0

      vram = o.get("memoryInGb") or o.get("gpu_memory") or o.get("ramGb") or o.get("vram") or 0
      try:
        vram = float(vram)
        if vram >= 1024:
          vram = vram / 1024.0
      except (ValueError, TypeError):
        vram = 0

      if gname not in gpu_aggregates:
        gpu_aggregates[gname] = {
            "id": gname,
            "displayName": gname,
            "memoryInGb": int(vram),
            "minPrice": price,
            "maxPrice": price,
            "count": 1
        }
      else:
        agg = gpu_aggregates[gname]
        agg["minPrice"] = min(agg["minPrice"], price)
        agg["maxPrice"] = max(agg["maxPrice"], price)
        agg["count"] += 1
        if not agg["memoryInGb"] and vram:
          agg["memoryInGb"] = int(vram)

    result = list(gpu_aggregates.values())
    if filter_pattern:
      regex = re.compile(filter_pattern, re.IGNORECASE)
      result = [
          g for g in result
          if regex.search(g.get("displayName", "")) or regex.search(g.get("id", ""))
      ]
    return result
