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
