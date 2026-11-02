from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


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
  def list_gpus(self, filter_pattern: Optional[str] = None, expand: bool = False) -> List[Dict[str, Any]]:
    pass

  def list_templates(self, filter_pattern: Optional[str] = None) -> List[Template]:
    """List available templates or images for this provider."""
    return []
