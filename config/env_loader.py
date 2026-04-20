"""
Environment variable loader utility.
Reads variables from .env file and provides them to other modules.
"""
import os
from pathlib import Path
from typing import Optional, Union, Any


class EnvConfig:
    """Configuration loaded from .env file with automatic type deduction"""

    def __init__(self, env_path: Optional[str] = None):
        """
        Load environment variables from .env file.

        Args:
            env_path: Path to .env file. If None, looks for .env in project root.
        """
        if env_path is None:
            # Default to project root
            project_root = Path(__file__).parent.parent
            env_path = project_root / '.env'

        self._config = {}
        self._load_env(env_path)

    def _load_env(self, env_path: Path):
        """Parse and load .env file, falling back to existing os.environ if file is absent."""
        if not os.path.exists(env_path):
            return  # Cloud Run supplies config via env vars; no .env file needed

        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue

                # Parse KEY=VALUE
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()

                    # Deduce type and store
                    typed_value = self._deduce_type(value)
                    self._config[key] = typed_value

                    # Set as environment variable (keep as string)
                    os.environ[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key with optional default."""
        return self._config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return self._config.get(name)

    # Anthropic API configuration
    @property
    def anthropic_base_url(self) -> str:
        return getattr(self, 'ANTHROPIC_BASE_URL', '')

    @property
    def anthropic_auth_token(self) -> str:
        return getattr(self, 'ANTHROPIC_AUTH_TOKEN', '')

    @property
    def anthropic_model(self) -> str:
        return getattr(self, 'ANTHROPIC_MODEL', 'claude-sonnet-4-6')

    # Model configuration
    @property
    def num_of_look_ahead(self) -> int:
        value = getattr(self, 'NUM_OF_LOOK_AHEAD', '10')
        return int(value)

    @property
    def use_last_n(self):
        value = getattr(self, 'USE_LAST_N', 'None')
        return None if value == 'None' else int(value)

    @property
    def normalize(self) -> bool:
        value = getattr(self, 'NORMALIZE', 'True')
        return value.lower() in ('true', '1', 'yes')

    @property
    def seq_len(self) -> int:
        value = getattr(self, 'SEQ_LEN', '64')
        return int(value)

    @property
    def csv_path(self) -> str:
        return getattr(self, 'CSV_PATH', 'data/Binance_BTC_USDT_USDT_3m.csv')

    @property
    def backtesting_csv_path(self) -> str:
        return getattr(self, 'BACKTESTING_CSV_PATH', 'data/Binance_BTC_USDT_USDT_3m.csv')

    @property
    def sliding_step(self) -> int:
        value = getattr(self, 'SLIDING_STEP', '1')
        return int(value)

    @property
    def pred_len(self) -> int:
        value = getattr(self, 'PRED_LEN', '1')
        return int(value)

    @property
    def sliding_window(self) -> bool:
        value = getattr(self, 'SLIDING_WINDOW', 'False')
        return value.lower() in ('true', '1', 'yes')

    @property
    def confidence_threshold(self) -> float:
        value = getattr(self, 'CONFIDENCE_THRESHOLD', '0.6')
        return float(value)

    @property
    def take_profit(self) -> float:
        value = getattr(self, 'TAKE_PROFIT', '125')
        return float(value)

    @property
    def margin(self) -> float:
        value = getattr(self, 'TMARGIN', '10')
        return float(value)

    @property
    def take_loss(self) -> float:
        value = getattr(self, 'TAKKE_LOSS', '50')
        return float(value)


# Global config instance
_config: Optional[EnvConfig] = None


def get_config() -> EnvConfig:
    """Get or create global config instance"""
    global _config
    if _config is None:
        _config = EnvConfig()
    return _config


def reload_config(env_path: Optional[str] = None):
    """Reload configuration from .env file"""
    global _config
    _config = EnvConfig(env_path)
    return _config
