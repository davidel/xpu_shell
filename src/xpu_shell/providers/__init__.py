import os
from pathlib import Path
from typing import Optional, Tuple

from xpu_shell.providers.base import BaseProvider, Instance
from xpu_shell.providers.runpod_provider import RunPodProvider
from xpu_shell.providers.vast_provider import VastProvider
from xpu_shell.providers.spheron_provider import SpheronProvider


CONFIG_DIR = Path.home() / ".config" / "xpu_shell"
LAST_PROVIDER_FILE = CONFIG_DIR / ".last_provider"


def get_last_provider_file() -> Path:
  custom_dir = os.environ.get("XPU_SHELL_CONFIG_DIR") or os.environ.get("RUNPOD_SHELL_CONFIG_DIR")
  if custom_dir:
    return Path(custom_dir) / ".last_provider"
  return LAST_PROVIDER_FILE



def save_last_provider(provider_name: str) -> None:
  try:
    target_file = get_last_provider_file()
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text(str(provider_name).strip().lower())
  except Exception:
    pass


def get_last_provider() -> Optional[str]:
  try:
    target_file = get_last_provider_file()
    if target_file.exists():
      content = target_file.read_text().strip().lower()
      if content:
        return content
  except Exception:
    pass
  return None


def get_provider(name: str = "runpod", api_key: Optional[str] = None) -> BaseProvider:
  norm = (name or "runpod").strip().lower()
  if norm in ("runpod", "rp"):
    return RunPodProvider(api_key=api_key)
  elif norm in ("vast", "vastai", "vast.ai"):
    return VastProvider(api_key=api_key)
  elif norm in ("spheron", "spheron-ai", "spheron.network"):
    return SpheronProvider(api_key=api_key)
  else:
    raise ValueError(f"Unknown provider '{name}'. Supported providers: 'runpod', 'vast', 'spheron'.")


def parse_instance_identifier(raw_id: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
  if not raw_id:
    return None, None
  raw_id = str(raw_id).strip()
  if ":" in raw_id:
    prov, pid = raw_id.split(":", 1)
    if prov.lower() in ("runpod", "rp", "vast", "vastai", "spheron"):
      return prov.lower(), pid
  return None, raw_id


def resolve_provider(args=None, instance_id: Optional[str] = None) -> BaseProvider:
  # 1. Explicit CLI argument
  cli_provider = getattr(args, "provider", None) if args else None
  if cli_provider:
    api_key = getattr(args, "api_key", None) if args else None
    return get_provider(cli_provider, api_key=api_key)

  # 2. Provider-specific CLI arguments
  if getattr(args, "ssh_key_id", None) is not None:
    api_key = getattr(args, "api_key", None) if args else None
    return get_provider("spheron", api_key=api_key)

  if getattr(args, "offer_id", None) is not None:
    api_key = getattr(args, "api_key", None) if args else None
    return get_provider("vast", api_key=api_key)

  # 3. Prefixed instance identifier e.g. "vast:12345" or "spheron:xyz"
  if instance_id:
    prov, _ = parse_instance_identifier(instance_id)
    if prov:
      return get_provider(prov)

  # 4. Spheron ID prefix heuristic
  if instance_id and instance_id.lower().startswith("spheron-"):
    return get_provider("spheron")

  # 5. Numeric ID heuristic for Vast.ai (RunPod IDs are alphanumeric/UUIDs)
  if instance_id and instance_id.isdigit():
    return get_provider("vast")

  # 4. Last used provider from state file
  last_prov = get_last_provider()
  if last_prov:
    return get_provider(last_prov)

  # 5. Environment variable
  env_prov = os.environ.get("XPU_SHELL_PROVIDER") or os.environ.get("DEFAULT_PROVIDER")
  if env_prov:
    return get_provider(env_prov)

  # 6. Default fallback
  return get_provider("runpod")
