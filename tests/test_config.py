import pytest
from app.core.config import Settings


def test_settings_default_values():
    settings = Settings()
    
    assert settings.app_name == "FX自動売買システム"
    assert settings.debug == True
    assert settings.database_url == "sqlite:///./data/fx_automation.db"
    assert settings.default_symbol == "USDJPY"
    assert settings.risk_per_trade == 0.02
    assert settings.api_v1_str == "/api/v1"


def test_settings_environment_override():
    import os
    os.environ["DEFAULT_SYMBOL"] = "EURJPY"
    os.environ["RISK_PER_TRADE"] = "0.01"
    
    settings = Settings()
    
    assert settings.default_symbol == "EURJPY"
    assert settings.risk_per_trade == 0.01
    
    # Clean up
    del os.environ["DEFAULT_SYMBOL"]
    del os.environ["RISK_PER_TRADE"]