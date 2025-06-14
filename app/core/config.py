from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    app_name: str = "FX自動売買システム"
    debug: bool = True
    
    # Database
    database_url: str = "sqlite:///./data/fx_automation.db"
    
    # MT5 Settings
    mt5_login: Optional[int] = None
    mt5_password: Optional[str] = None
    mt5_server: Optional[str] = None
    
    # Trading Settings
    default_symbol: str = "USDJPY"
    risk_per_trade: float = 0.02  # 2%
    
    # API Settings
    api_v1_str: str = "/api/v1"


settings = Settings()