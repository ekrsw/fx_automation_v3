import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.historical_data_service import (
    HistoricalDataService, DataQualityStatus, DataQualityReport, HistoricalDataSummary
)
from app.models.price_data import PriceData, Base


@pytest.fixture
def db_session():
    """テスト用データベースセッション"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def mock_mt5_service():
    """モックMT5サービス"""
    with patch('app.services.historical_data_service.MT5Service') as mock:
        instance = mock.return_value
        instance.connect.return_value = True
        instance.disconnect.return_value = None
        yield instance


@pytest.fixture
def historical_service(db_session, mock_mt5_service):
    """履歴データサービス"""
    return HistoricalDataService(db_session)


@pytest.fixture
def sample_price_data():
    """サンプル価格データ"""
    dates = [datetime.now() - timedelta(hours=i) for i in range(100, 0, -1)]
    data = []
    
    for i, date in enumerate(dates):
        price = 150.0 + (i % 10) * 0.1
        data.append({
            'datetime': date,
            'open': price,
            'high': price + 0.05,
            'low': price - 0.05,
            'close': price + 0.02,
            'volume': 1000 + i
        })
    
    return pd.DataFrame(data)


def test_historical_data_service_initialization(historical_service):
    """履歴データサービス初期化テスト"""
    assert historical_service.db is not None
    assert historical_service.mt5_service is not None
    assert len(historical_service.supported_symbols) > 0
    assert len(historical_service.supported_timeframes) > 0
    assert 'USDJPY' in historical_service.supported_symbols
    assert 'H1' in historical_service.supported_timeframes


def test_normalize_data(historical_service, sample_price_data):
    """データ正規化テスト"""
    normalized = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    
    # 必要カラムの存在確認
    required_columns = ['symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close', 'volume']
    for col in required_columns:
        assert col in normalized.columns
    
    # データ型確認
    assert normalized['symbol'].iloc[0] == "USDJPY"
    assert normalized['timeframe'].iloc[0] == "H1"
    assert pd.api.types.is_datetime64_any_dtype(normalized['datetime'])
    assert pd.api.types.is_numeric_dtype(normalized['open'])
    assert pd.api.types.is_numeric_dtype(normalized['volume'])
    
    # レコード数確認
    assert len(normalized) <= len(sample_price_data)  # 無効データが除去される可能性


def test_validate_price_data(historical_service):
    """価格データ妥当性テストt"""
    # 正常データ
    valid_data = pd.DataFrame([
        {
            'datetime': datetime.now(),
            'open': 150.0,
            'high': 150.5,
            'low': 149.5,
            'close': 150.2,
            'volume': 1000
        }
    ])
    
    validated = historical_service._validate_price_data(valid_data)
    assert len(validated) == 1
    
    # 異常データ（high < low）
    invalid_data = pd.DataFrame([
        {
            'datetime': datetime.now(),
            'open': 150.0,
            'high': 149.0,  # high < low
            'low': 150.0,
            'close': 149.5,
            'volume': 1000
        }
    ])
    
    validated = historical_service._validate_price_data(invalid_data)
    assert len(validated) == 0  # 無効データは除去される


def test_check_data_quality_good_data(historical_service, sample_price_data):
    """データ品質チェック - 良好データ"""
    normalized_data = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    quality_report = historical_service._check_data_quality(normalized_data, "USDJPY", "H1")
    
    assert isinstance(quality_report, DataQualityReport)
    assert quality_report.symbol == "USDJPY"
    assert quality_report.timeframe == "H1"
    assert quality_report.total_records > 0
    assert quality_report.quality_score >= 0
    assert quality_report.quality_score <= 100
    assert quality_report.status in [status.value for status in DataQualityStatus]


def test_check_data_quality_empty_data(historical_service):
    """データ品質チェック - 空データ"""
    empty_data = pd.DataFrame()
    quality_report = historical_service._check_data_quality(empty_data, "USDJPY", "H1")
    
    assert quality_report.status == DataQualityStatus.MISSING_DATA
    assert quality_report.total_records == 0
    assert quality_report.quality_score == 0.0
    assert len(quality_report.issues) > 0


def test_count_time_gaps(historical_service):
    """時間ギャップカウントテスト"""
    # 連続データ（ギャップなし）
    continuous_data = pd.DataFrame([
        {'datetime': datetime(2023, 1, 1, i, 0, 0)} for i in range(10)
    ])
    
    gaps = historical_service._count_time_gaps(continuous_data, "H1")
    assert gaps == 0
    
    # ギャップありデータ
    gap_data = pd.DataFrame([
        {'datetime': datetime(2023, 1, 1, 0, 0, 0)},
        {'datetime': datetime(2023, 1, 1, 1, 0, 0)},
        {'datetime': datetime(2023, 1, 1, 5, 0, 0)},  # 3時間のギャップ
        {'datetime': datetime(2023, 1, 1, 6, 0, 0)},
    ])
    
    gaps = historical_service._count_time_gaps(gap_data, "H1")
    assert gaps >= 1


def test_clean_data(historical_service, sample_price_data):
    """データクリーニングテスト"""
    # 重複データを追加
    dirty_data = sample_price_data.copy()
    dirty_data = pd.concat([dirty_data, dirty_data.iloc[:5]], ignore_index=True)
    
    # 品質レポート作成（重複ありの状態）
    quality_report = DataQualityReport(
        symbol="USDJPY",
        timeframe="H1",
        total_records=len(dirty_data),
        date_range=(dirty_data['datetime'].min(), dirty_data['datetime'].max()),
        gaps_count=0,
        duplicates_count=5,
        invalid_records=0,
        quality_score=80.0,
        status=DataQualityStatus.HAS_DUPLICATES,
        issues=[],
        recommendations=[]
    )
    
    cleaned_data = historical_service._clean_data(dirty_data, quality_report)
    
    # 重複が除去されているか確認
    assert len(cleaned_data) < len(dirty_data)
    assert cleaned_data['datetime'].is_monotonic_increasing  # 時間順ソート確認


def test_get_existing_data_range_no_data(historical_service):
    """既存データ範囲取得 - データなし"""
    date_range = historical_service._get_existing_data_range("NONEXISTENT", "H1")
    assert date_range is None


def test_get_existing_data_range_with_data(historical_service, sample_price_data):
    """既存データ範囲取得 - データあり"""
    # テストデータをDBに保存
    normalized_data = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    historical_service._save_to_database(normalized_data)
    
    date_range = historical_service._get_existing_data_range("USDJPY", "H1")
    assert date_range is not None
    assert len(date_range) == 2
    assert date_range[0] <= date_range[1]


def test_save_to_database(historical_service, sample_price_data):
    """データベース保存テスト"""
    normalized_data = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    
    saved_count = historical_service._save_to_database(normalized_data)
    assert saved_count > 0
    assert saved_count <= len(normalized_data)
    
    # データベースから確認
    db_count = historical_service.db.query(PriceData).filter(
        PriceData.symbol == "USDJPY",
        PriceData.timeframe == "H1"
    ).count()
    assert db_count == saved_count


def test_save_to_database_duplicates(historical_service, sample_price_data):
    """データベース保存 - 重複処理テスト"""
    normalized_data = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    
    # 初回保存
    first_save = historical_service._save_to_database(normalized_data)
    
    # 同じデータを再保存（重複スキップ）
    second_save = historical_service._save_to_database(normalized_data)
    assert second_save == 0  # 重複データはスキップされる
    
    # 強制更新
    force_update_save = historical_service._save_to_database(normalized_data, force_update=True)
    assert force_update_save == first_save  # 全データが更新される


def test_get_data_overview(historical_service, sample_price_data):
    """データ概要取得テスト"""
    # テストデータ保存
    normalized_data = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    historical_service._save_to_database(normalized_data)
    
    overview = historical_service.get_data_overview()
    
    assert isinstance(overview, dict)
    if overview:  # データがある場合
        assert "USDJPY" in overview
        assert "H1" in overview["USDJPY"]
        assert "count" in overview["USDJPY"]["H1"]
        assert overview["USDJPY"]["H1"]["count"] > 0


def test_validate_historical_data_existing(historical_service, sample_price_data):
    """保存済み履歴データ検証テスト"""
    # テストデータ保存
    normalized_data = historical_service._normalize_data(sample_price_data, "USDJPY", "H1")
    historical_service._save_to_database(normalized_data)
    
    quality_report = historical_service.validate_historical_data("USDJPY", "H1")
    
    assert isinstance(quality_report, DataQualityReport)
    assert quality_report.symbol == "USDJPY"
    assert quality_report.timeframe == "H1"
    assert quality_report.total_records > 0
    assert quality_report.status != DataQualityStatus.MISSING_DATA


def test_validate_historical_data_nonexistent(historical_service):
    """存在しないデータの検証テスト"""
    quality_report = historical_service.validate_historical_data("NONEXISTENT", "H1")
    
    assert quality_report.status == DataQualityStatus.MISSING_DATA
    assert quality_report.total_records == 0
    assert len(quality_report.issues) > 0


def test_fetch_and_store_historical_data_success(historical_service, sample_price_data):
    """履歴データ取得・保存成功テスト"""
    # _fetch_mt5_dataメソッドをモック
    with patch.object(historical_service, '_fetch_mt5_data') as mock_fetch:
        mock_fetch.return_value = sample_price_data
        
        # 履歴データ取得実行
        summary = historical_service.fetch_and_store_historical_data(
            symbol="USDJPY",
            timeframe="H1",
            years_back=1
        )
        
        assert isinstance(summary, HistoricalDataSummary)
        assert summary.symbol == "USDJPY"
        assert summary.timeframe == "H1"
        assert summary.records_fetched > 0
        assert summary.records_saved > 0
        assert summary.processing_time > 0
        assert summary.quality_report is not None


def test_fetch_and_store_historical_data_mt5_failure(historical_service):
    """履歴データ取得 - MT5接続失敗テスト"""
    # MT5接続失敗のモック
    with patch.object(historical_service.mt5_service, 'connect') as mock_connect:
        mock_connect.return_value = False
        
        with pytest.raises(Exception, match="MT5接続に失敗しました"):
            historical_service.fetch_and_store_historical_data(
                symbol="USDJPY",
                timeframe="H1",
                years_back=1
            )


def test_fetch_and_store_historical_data_no_data(historical_service):
    """履歴データ取得 - データなしテスト"""
    # データなしのモック
    with patch.object(historical_service, '_fetch_mt5_data') as mock_fetch:
        mock_fetch.return_value = None
        
        summary = historical_service.fetch_and_store_historical_data(
            symbol="USDJPY",
            timeframe="H1",
            years_back=1
        )
        
        assert summary.records_fetched == 0
        assert summary.records_saved == 0
        assert summary.quality_report.status == DataQualityStatus.MISSING_DATA


def test_quality_thresholds(historical_service):
    """品質閾値テスト"""
    thresholds = historical_service.quality_thresholds
    
    assert 'max_gap_ratio' in thresholds
    assert 'max_duplicate_ratio' in thresholds
    assert 'min_completeness' in thresholds
    
    assert 0 < thresholds['max_gap_ratio'] < 1
    assert 0 < thresholds['max_duplicate_ratio'] < 1
    assert 0 < thresholds['min_completeness'] <= 1


def test_supported_symbols_and_timeframes(historical_service):
    """サポート通貨ペア・タイムフレームテスト"""
    assert len(historical_service.supported_symbols) >= 5
    assert len(historical_service.supported_timeframes) >= 5
    
    # 主要通貨ペアの存在確認
    major_pairs = ["USDJPY", "EURUSD", "GBPUSD"]
    for pair in major_pairs:
        assert pair in historical_service.supported_symbols
    
    # 主要タイムフレームの存在確認
    major_timeframes = ["M1", "M5", "H1", "D1"]
    for tf in major_timeframes:
        assert tf in historical_service.supported_timeframes


def test_error_handling_in_data_processing(historical_service):
    """データ処理エラーハンドリングテスト"""
    # 無効なデータフレーム
    invalid_data = pd.DataFrame([{'invalid': 'data'}])
    
    with pytest.raises(ValueError):
        historical_service._normalize_data(invalid_data, "USDJPY", "H1")


def test_large_dataset_handling(historical_service):
    """大量データ処理テスト"""
    # 大量のサンプルデータ作成（1000件）
    dates = [datetime.now() - timedelta(hours=i) for i in range(1000, 0, -1)]
    large_data = pd.DataFrame([
        {
            'datetime': date,
            'open': 150.0 + (i % 100) * 0.01,
            'high': 150.0 + (i % 100) * 0.01 + 0.05,
            'low': 150.0 + (i % 100) * 0.01 - 0.05,
            'close': 150.0 + (i % 100) * 0.01 + 0.02,
            'volume': 1000 + i
        }
        for i, date in enumerate(dates)
    ])
    
    # 正規化テスト
    normalized = historical_service._normalize_data(large_data, "USDJPY", "H1")
    assert len(normalized) > 900  # 大部分のデータが保持される
    
    # 品質チェックテスト
    quality_report = historical_service._check_data_quality(normalized, "USDJPY", "H1")
    assert quality_report.total_records > 900
    assert quality_report.quality_score > 50  # 合理的な品質スコア