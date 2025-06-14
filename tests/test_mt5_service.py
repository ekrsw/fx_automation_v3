import pytest
from unittest.mock import Mock, patch
import pandas as pd
from datetime import datetime
from app.services.mt5_service import MT5Service


@pytest.fixture
def mt5_service():
    return MT5Service()


@patch('app.services.mt5_service.mt5')
def test_connect_success(mock_mt5, mt5_service):
    mock_mt5.initialize.return_value = True
    mock_mt5.login.return_value = True
    
    result = mt5_service.connect()
    
    assert result == True
    assert mt5_service.connected == True
    mock_mt5.initialize.assert_called_once()


@patch('app.services.mt5_service.mt5')
def test_connect_failure(mock_mt5, mt5_service):
    mock_mt5.initialize.return_value = False
    mock_mt5.last_error.return_value = (1, "Connection failed")
    
    result = mt5_service.connect()
    
    assert result == False
    assert mt5_service.connected == False


@patch('app.services.mt5_service.mt5')
def test_disconnect(mock_mt5, mt5_service):
    mt5_service.connected = True
    
    mt5_service.disconnect()
    
    assert mt5_service.connected == False
    mock_mt5.shutdown.assert_called_once()


@patch('app.services.mt5_service.mt5')
def test_is_connected(mock_mt5, mt5_service):
    mt5_service.connected = True
    mock_mt5.terminal_info.return_value = Mock()
    
    result = mt5_service.is_connected()
    
    assert result == True


@patch('app.services.mt5_service.mt5')
def test_get_price_data_success(mock_mt5, mt5_service):
    mt5_service.connected = True
    mock_mt5.terminal_info.return_value = Mock()
    
    # モックデータ
    mock_rates = [
        {
            'time': 1640995200,  # 2022-01-01 00:00:00 UTC
            'open': 110.0,
            'high': 110.5,
            'low': 109.5,
            'close': 110.2,
            'tick_volume': 1000
        }
    ]
    mock_mt5.copy_rates_range.return_value = mock_rates
    mock_mt5.TIMEFRAME_M1 = 1
    
    result = mt5_service.get_price_data("USDJPY", "M1", 100)
    
    assert result is not None
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert 'datetime' in result.columns
    assert 'open' in result.columns


@patch('app.services.mt5_service.mt5')
def test_get_price_data_no_connection(mock_mt5, mt5_service):
    mt5_service.connected = False
    
    result = mt5_service.get_price_data("USDJPY", "M1", 100)
    
    assert result is None


@patch('app.services.mt5_service.mt5')
def test_get_symbol_info(mock_mt5, mt5_service):
    mt5_service.connected = True
    mock_mt5.terminal_info.return_value = Mock()
    
    mock_info = Mock()
    mock_info.bid = 110.0
    mock_info.ask = 110.1
    mock_info.spread = 10
    mock_info.digits = 3
    mock_info.point = 0.001
    mock_mt5.symbol_info.return_value = mock_info
    
    result = mt5_service.get_symbol_info("USDJPY")
    
    assert result is not None
    assert result["symbol"] == "USDJPY"
    assert result["bid"] == 110.0
    assert result["ask"] == 110.1


@patch('app.services.mt5_service.mt5')
def test_get_account_info(mock_mt5, mt5_service):
    mt5_service.connected = True
    mock_mt5.terminal_info.return_value = Mock()
    
    mock_info = Mock()
    mock_info.login = 12345
    mock_info.balance = 10000.0
    mock_info.equity = 10500.0
    mock_info.profit = 500.0
    mock_info.margin = 1000.0
    mock_info.currency = "USD"
    mock_mt5.account_info.return_value = mock_info
    
    result = mt5_service.get_account_info()
    
    assert result is not None
    assert result["login"] == 12345
    assert result["balance"] == 10000.0
    assert result["currency"] == "USD"