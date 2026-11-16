from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class Instance:
  id: str
  name: str
  provider: str
  status: str
  ssh_host: Optional[str] = None
  ssh_port: Optional[int] = None
  gpu_name: Optional[str] = None
  gpu_count: int = 1
  uptime_seconds: Optional[int] = None
  cost_per_hr: Optional[float] = None
  vcpu_count: Optional[int] = None
  memory_gb: Optional[float] = None
  disk_gb: Optional[float] = None
  image_name: Optional[str] = None
  raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Template:
  id: str
  name: str
  provider: str
  type: str = "template"
  image_name: Optional[str] = None
  description: Optional[str] = None
  raw: Dict[str, Any] = field(default_factory=dict)


def format_template_identifier(kind: str, value: str) -> str:
  clean_kind = kind.lower().strip()
  clean_val = value.strip()
  if clean_val.startswith(f"{clean_kind}:"):
    return clean_val
  return f"{clean_kind}:{clean_val}"


def parse_template_identifier(spec: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
  """Parse a template or image specifier into (kind, value).

  Kind will be either 'template' or 'image' (or None if spec is None).
  Examples:
    'image:pytorch/pytorch:latest' -> ('image', 'pytorch/pytorch:latest')
    'template:c21f8a' -> ('template', 'c21f8a')
    'pytorch/pytorch:latest' -> ('image', 'pytorch/pytorch:latest')
    'c21f8a' -> ('template', 'c21f8a')
  """
  if not spec:
    return None, None
  spec = spec.strip()
  if spec.startswith("image:"):
    return "image", spec[len("image:"):].strip()
  if spec.startswith("template:"):
    return "template", spec[len("template:"):].strip()
  if "/" in spec or ":" in spec or spec.lower().startswith("ubuntu") or spec.lower().startswith("debian"):
    return "image", spec
  return "template", spec


def parse_ssh_command(cmd: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
  """Parse an SSH command line invocation into (host, port)."""
  if not cmd or not isinstance(cmd, str):
    return None, None

  host = None
  port = None

  port_m = re.search(r"-p\s*(\d+)", cmd)
  if port_m:
    try:
      port = int(port_m.group(1))
    except (ValueError, TypeError):
      port = None

  cleaned = re.sub(r"-[a-zA-Z]\s+[^\s]+", "", cmd)
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

  if host and port is None:
    port = 22

  return host, port


def parse_duration_hours(val: Any) -> float:
  """Parse a duration string or number into hours as a float.

  Supported suffixes: 'h' (hours), 'd' (days), 'w' (weeks), 'm' (minutes).
  Unadorned numbers are treated as hours.
  Examples: '24h' -> 24.0, '7d' -> 168.0, '2w' -> 336.0, 48 -> 48.0
  """
  if val is None:
    raise ValueError("Duration cannot be None.")

  if isinstance(val, (int, float)):
    hours = float(val)
    if hours <= 0:
      raise ValueError(f"Duration must be greater than 0, got {val}.")
    return hours

  s = str(val).strip().lower()
  if not s:
    raise ValueError("Duration cannot be empty.")

  multiplier = 1.0
  num_part = s

  if s.endswith("weeks") or s.endswith("week") or s.endswith("w"):
    multiplier = 24.0 * 7.0
    num_part = re.sub(r"(weeks?|w)$", "", s).strip()
  elif s.endswith("days") or s.endswith("day") or s.endswith("d"):
    multiplier = 24.0
    num_part = re.sub(r"(days?|d)$", "", s).strip()
  elif s.endswith("hours") or s.endswith("hour") or s.endswith("h"):
    multiplier = 1.0
    num_part = re.sub(r"(hours?|h)$", "", s).strip()
  elif s.endswith("minutes") or s.endswith("minute") or s.endswith("m"):
    multiplier = 1.0 / 60.0
    num_part = re.sub(r"(minutes?|m)$", "", s).strip()

  try:
    val_num = float(num_part)
  except ValueError:
    raise ValueError(f"Invalid duration format: '{val}'. Expected format like '24h', '7d', '2w'.")

  if val_num <= 0:
    raise ValueError(f"Duration must be greater than 0, got '{val}'.")

  return val_num * multiplier


def format_duration_hours(hours: Optional[float]) -> str:
  """Format duration in hours for display in table columns."""
  if hours is None:
    return "N/A"
  if hours <= 0:
    return "Expired"
  if hours >= 10:
    return f"{int(round(hours))}h"
  return f"{hours:.1f}h"


def format_expiration(raw_end_date: Any, now_ts: Optional[float] = None) -> Optional[str]:
  """Format expiration timestamp into a readable date and remaining hours.

  Returns None if raw_end_date is None or represents a distant future (> 5 years).
  """
  if not raw_end_date:
    return None

  end_ts = None
  if isinstance(raw_end_date, (int, float)):
    end_ts = float(raw_end_date)
  elif isinstance(raw_end_date, str):
    s = raw_end_date.strip()
    try:
      end_ts = float(s)
    except ValueError:
      try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        end_ts = dt.timestamp()
      except Exception:
        return None
  else:
    return None

  if end_ts < 1e8:
    return None

  if now_ts is None:
    now_ts = time.time()

  rem_hours = (end_ts - now_ts) / 3600.0
  if rem_hours > 43800:
    return None

  try:
    dt = datetime.fromtimestamp(end_ts, timezone.utc)
    date_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
  except Exception:
    return None

  if rem_hours > 0:
    if rem_hours >= 10:
      rem_str = f"{int(round(rem_hours))} hours remaining"
    elif rem_hours >= 1:
      rem_str = f"{rem_hours:.1f} hours remaining"
    else:
      rem_mins = max(1, int(round(rem_hours * 60)))
      rem_str = f"{rem_mins} minutes remaining"
    return f"{date_str} ({rem_str})"
  else:
    return f"{date_str} (expired)"


class BaseProvider(ABC):

  @property
  @abstractmethod
  def name(self) -> str:
    pass

  @property
  def default_image(self) -> Optional[str]:
    """Default container/OS image for this provider, if any."""
    return None

  def register_create_args(self, parser: Any) -> None:
    """Register provider-specific CLI arguments for the create command."""
    pass

  def register_commands(self, subparsers: Any) -> None:
    """Register provider-specific subcommands."""
    pass

  @abstractmethod
  def list_instances(self) -> List[Instance]:
    pass

  @abstractmethod
  def get_instance(self, instance_id: str) -> Instance:
    pass

  @abstractmethod
  def get_ssh_endpoint(self, instance_id: str) -> Tuple[Optional[str], Optional[int]]:
    pass

  @abstractmethod
  def create_instance(self, **kwargs) -> Instance:
    pass

  @abstractmethod
  def stop_instance(self, instance_id: str) -> None:
    pass

  @abstractmethod
  def start_instance(self, instance_id: str) -> None:
    pass

  @abstractmethod
  def terminate_instance(self, instance_id: str) -> None:
    pass

  @abstractmethod
  def list_gpus(self, filter_pattern: Optional[str] = None, expand: bool = False, **kwargs) -> List[Dict[str, Any]]:
    pass

  def list_templates(self, filter_pattern: Optional[str] = None) -> List[Template]:
    """List available templates or images for this provider."""
    return []
