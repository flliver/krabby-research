"""HAL client package."""
# Re-export commonly used items for cleaner imports
from hal.client.client import HalClient
from hal.client.config import HalClientConfig

__all__ = ["HalClient", "HalClientConfig"]

