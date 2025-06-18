import pytest
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.core.database import get_db
from app.models.price_data import Base
from app.services.historical_data_service import HistoricalDataSummary, DataQualityReport, DataQualityStatus
from app.services.backtest_engine import BacktestResult, BacktestConfig, BacktestMetrics, BacktestStatus
from app.services.performance_analyzer import ComprehensiveAnalysis, PerformanceScore, PerformanceRating
from app.services.report_generator import GeneratedReport, ReportConfig, ReportFormat


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
def client(db_session):
    """テストクライアント"""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_historical_data_request():
    """サンプル履歴データリクエスト"""
    return {
        "symbol": "USDJPY",
        "timeframe": "H1",
        "years_back": 2,
        "force_update": False
    }


@pytest.fixture
def sample_backtest_request():
    """サンプルバックテストリクエスト"""
    return {
        "symbol": "USDJPY",
        "timeframe": "H1",
        "start_date": (datetime.now() - timedelta(days=90)).isoformat(),
        "end_date": datetime.now().isoformat(),
        "initial_balance": 100000.0,
        "risk_per_trade": 0.02,
        "commission_per_lot": 5.0,
        "spread_pips": 2.0,
        "min_signal_confidence": 0.7,
        "min_risk_reward": 1.5,
        "max_positions": 3
    }


@pytest.fixture
def sample_report_request():
    """サンプルレポートリクエスト"""
    return {
        "report_type": "single_strategy",
        "format": "html",
        "title": "テストレポート",
        "include_charts": True,
        "include_detailed_trades": False
    }


def test_fetch_historical_data_endpoint(client, sample_historical_data_request):
    """履歴データ取得エンドポイントテスト"""
    with patch('app.api.api_v1.endpoints.backtest.HistoricalDataService') as mock_service:
        mock_instance = mock_service.return_value
        
        response = client.post(
            "/api/v1/backtest/historical-data/fetch",
            json=sample_historical_data_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["symbol"] == "USDJPY"
        assert data["timeframe"] == "H1"
        assert "履歴データ取得を開始しました" in data["message"]


def test_fetch_all_historical_data_endpoint(client):
    """全履歴データ取得エンドポイントテスト"""
    with patch('app.api.api_v1.endpoints.backtest.HistoricalDataService') as mock_service:
        mock_instance = mock_service.return_value
        mock_instance.supported_symbols = ["USDJPY", "EURUSD", "GBPUSD"]
        
        response = client.post(
            "/api/v1/backtest/historical-data/fetch-all",
            params={
                "timeframe": "H1",
                "years_back": 3,
                "force_update": False
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert data["timeframe"] == "H1"
        assert data["years_back"] == 3
        assert data["symbols_count"] == 3


def test_get_data_overview_endpoint(client):
    """データ概要取得エンドポイントテスト"""
    mock_overview = {
        "USDJPY": {
            "H1": {"count": 1000, "start_date": "2023-01-01", "end_date": "2023-12-31"}
        },
        "EURUSD": {
            "H1": {"count": 950, "start_date": "2023-01-01", "end_date": "2023-12-31"}
        }
    }
    
    with patch('app.api.api_v1.endpoints.backtest.HistoricalDataService') as mock_service:
        mock_instance = mock_service.return_value
        mock_instance.get_data_overview.return_value = mock_overview
        mock_instance.supported_timeframes = {"H1": 60, "H4": 240, "D1": 1440}
        mock_instance.supported_symbols = ["USDJPY", "EURUSD", "GBPUSD"]
        
        response = client.get("/api/v1/backtest/historical-data/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert "overview" in data
        assert "summary" in data
        assert data["summary"]["total_symbols"] == 2
        assert "USDJPY" in data["overview"]


def test_validate_data_quality_endpoint(client):
    """データ品質検証エンドポイントテスト"""
    mock_quality_report = DataQualityReport(
        symbol="USDJPY",
        timeframe="H1",
        total_records=1000,
        date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31)),
        gaps_count=5,
        duplicates_count=0,
        invalid_records=2,
        quality_score=92.5,
        status=DataQualityStatus.GOOD,
        issues=["小さなギャップが5箇所あります"],
        recommendations=["定期的なデータ更新を推奨"]
    )
    
    with patch('app.api.api_v1.endpoints.backtest.HistoricalDataService') as mock_service:
        mock_instance = mock_service.return_value
        mock_instance.validate_historical_data.return_value = mock_quality_report
        
        response = client.get("/api/v1/backtest/historical-data/quality/USDJPY/H1")
        
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "USDJPY"
        assert data["timeframe"] == "H1"
        assert data["total_records"] == 1000
        assert data["quality_score"] == 92.5
        assert data["status"] == "good"
        assert len(data["issues"]) == 1
        assert len(data["recommendations"]) == 1


def test_run_backtest_endpoint(client, sample_backtest_request):
    """バックテスト実行エンドポイントテスト"""
    mock_metrics = BacktestMetrics(
        total_trades=50,
        winning_trades=35,
        losing_trades=15,
        win_rate=0.7,
        total_profit=5000.0,
        gross_profit=8000.0,
        gross_loss=-3000.0,
        profit_factor=2.67,
        average_win=228.57,
        average_loss=-200.0,
        largest_win=500.0,
        largest_loss=-400.0,
        max_drawdown=0.08,
        max_consecutive_wins=8,
        max_consecutive_losses=3
    )
    
    mock_result = BacktestResult(
        config=BacktestConfig(
            symbol="USDJPY",
            timeframe="H1",
            start_date=datetime.now() - timedelta(days=90),
            end_date=datetime.now(),
            initial_balance=100000.0,
            risk_per_trade=0.02
        ),
        status=BacktestStatus.COMPLETED,
        metrics=mock_metrics,
        positions=[],
        equity_curve=pd.DataFrame(),
        daily_returns=pd.DataFrame(),
        execution_time=2.5
    )
    
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine:
        mock_instance = mock_engine.return_value
        mock_instance.run_backtest.return_value = mock_result
        
        response = client.post(
            "/api/v1/backtest/backtest/run",
            json=sample_backtest_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "COMPLETED"
        assert "backtest_id" in data
        assert "config" in data
        assert "metrics" in data
        assert data["metrics"]["total_trades"] == 50
        assert data["metrics"]["win_rate"] == 0.7
        assert data["execution_time"] == 2.5


def test_run_backtest_endpoint_failed(client, sample_backtest_request):
    """バックテスト実行失敗エンドポイントテスト"""
    mock_result = BacktestResult(
        config=None,
        status=BacktestStatus.FAILED,
        metrics=None,
        positions=[],
        equity_curve=pd.DataFrame(),
        daily_returns=pd.DataFrame(),
        execution_time=0.1,
        error_message="データが不足しています"
    )
    
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine:
        mock_instance = mock_engine.return_value
        mock_instance.run_backtest.return_value = mock_result
        
        response = client.post(
            "/api/v1/backtest/backtest/run",
            json=sample_backtest_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "FAILED"
        assert data["error_message"] == "データが不足しています"
        assert "metrics" not in data


def test_run_optimization_endpoint(client, sample_backtest_request):
    """最適化実行エンドポイントテスト"""
    optimization_request = {
        "base_config": sample_backtest_request,
        "parameter_ranges": {
            "risk_per_trade": [0.01, 0.02, 0.03],
            "min_signal_confidence": [0.6, 0.7, 0.8]
        }
    }
    
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine:
        mock_instance = mock_engine.return_value
        
        response = client.post(
            "/api/v1/backtest/backtest/optimize",
            json=optimization_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "started"
        assert "optimization_id" in data
        assert "parameter_ranges" in data
        assert "最適化を開始しました" in data["message"]


def test_analyze_performance_endpoint(client, sample_backtest_request):
    """パフォーマンス分析エンドポイントテスト"""
    mock_metrics = BacktestMetrics(
        total_trades=30,
        winning_trades=20,
        losing_trades=10,
        win_rate=0.67,
        total_profit=3000.0,
        profit_factor=2.0,
        max_drawdown=0.05
    )
    
    mock_result = BacktestResult(
        config=BacktestConfig(symbol="USDJPY", timeframe="H1"),
        status=BacktestStatus.COMPLETED,
        metrics=mock_metrics,
        positions=[],
        equity_curve=pd.DataFrame(),
        daily_returns=pd.DataFrame(),
        execution_time=1.5
    )
    
    mock_performance_score = PerformanceScore(
        overall_score=78.5,
        rating=PerformanceRating.GOOD,
        profitability_score=80.0,
        consistency_score=75.0,
        risk_management_score=82.0,
        efficiency_score=77.0,
        strengths=["高いプロフィットファクター", "良好なリスク管理"],
        weaknesses=["取引頻度が低い"],
        recommendations=["取引機会を増やす"]
    )
    
    mock_analysis = ComprehensiveAnalysis(
        backtest_result=mock_result,
        risk_metrics=Mock(),
        return_metrics=Mock(),
        ratio_metrics=Mock(),
        trading_metrics=Mock(),
        performance_score=mock_performance_score,
        executive_summary="良好な成績",
        detailed_analysis="詳細分析"
    )
    
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine, \
         patch('app.api.api_v1.endpoints.backtest.PerformanceAnalyzer') as mock_analyzer:
        
        mock_engine_instance = mock_engine.return_value
        mock_engine_instance.run_backtest.return_value = mock_result
        
        mock_analyzer_instance = mock_analyzer.return_value
        mock_analyzer_instance.analyze_performance.return_value = mock_analysis
        
        response = client.post(
            "/api/v1/backtest/analysis/performance",
            json=sample_backtest_request
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "analysis_id" in data
        assert "performance_score" in data
        assert data["performance_score"]["overall_score"] == 78.5
        assert data["performance_score"]["rating"] == "GOOD"
        assert "strengths" in data
        assert "recommendations" in data


def test_generate_report_endpoint(client, sample_backtest_request, sample_report_request):
    """レポート生成エンドポイントテスト"""
    mock_metrics = BacktestMetrics(
        total_trades=25,
        winning_trades=18,
        losing_trades=7,
        win_rate=0.72,
        total_profit=2500.0,
        profit_factor=2.5,
        max_drawdown=0.04
    )
    
    mock_result = BacktestResult(
        config=BacktestConfig(symbol="USDJPY", timeframe="H1"),
        status=BacktestStatus.COMPLETED,
        metrics=mock_metrics,
        positions=[],
        equity_curve=pd.DataFrame(),
        daily_returns=pd.DataFrame(),
        execution_time=1.8
    )
    
    mock_report = GeneratedReport(
        config=ReportConfig(
            report_type="single_strategy",
            format=ReportFormat.HTML,
            title="テストレポート"
        ),
        content="<html>Test Report Content</html>",
        file_path="reports/test_report.html",
        generation_time=datetime.now(),
        file_size=1024
    )
    
    request_body = {
        **sample_backtest_request,
        **sample_report_request
    }
    
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine, \
         patch('app.api.api_v1.endpoints.backtest.ReportGenerator') as mock_generator:
        
        mock_engine_instance = mock_engine.return_value
        mock_engine_instance.run_backtest.return_value = mock_result
        
        mock_generator_instance = mock_generator.return_value
        mock_generator_instance.generate_single_strategy_report.return_value = mock_report
        
        response = client.post(
            "/api/v1/backtest/reports/generate",
            json=request_body
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "report_id" in data
        assert data["status"] == "completed"
        assert data["file_path"] == "reports/test_report.html"
        assert data["file_size"] == 1024
        assert data["format"] == "html"
        assert "download_url" in data


def test_download_report_endpoint(client):
    """レポートダウンロードエンドポイントテスト"""
    # テスト用レポートファイル作成
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write("<html><body>Test Report</body></html>")
        temp_file_path = f.name
    
    filename = os.path.basename(temp_file_path)
    
    # reportsディレクトリとファイルをモック
    with patch('os.path.exists') as mock_exists, \
         patch('fastapi.responses.FileResponse') as mock_file_response:
        
        mock_exists.return_value = True
        mock_file_response.return_value = Mock()
        
        response = client.get(f"/api/v1/backtest/reports/download/{filename}")
        
        # FileResponseが呼ばれることを確認
        mock_file_response.assert_called_once()
    
    # テンポラリファイル削除
    os.unlink(temp_file_path)


def test_download_report_not_found(client):
    """レポートダウンロード - ファイル未発見"""
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        
        response = client.get("/api/v1/backtest/reports/download/nonexistent.html")
        
        assert response.status_code == 404
        data = response.json()
        assert "レポートファイルが見つかりません" in data["detail"]


def test_get_backtest_presets_endpoint(client):
    """バックテストプリセット取得エンドポイントテスト"""
    response = client.get("/api/v1/backtest/backtest/presets")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "timeframes" in data
    assert "symbols" in data
    assert "preset_configs" in data
    assert "optimization_ranges" in data
    
    # プリセット設定の確認
    assert "conservative" in data["preset_configs"]
    assert "moderate" in data["preset_configs"]
    assert "aggressive" in data["preset_configs"]
    
    # タイムフレーム・通貨ペアの確認
    assert "H1" in data["timeframes"]
    assert "USDJPY" in data["symbols"]


def test_get_system_status_endpoint(client):
    """システム状態取得エンドポイントテスト"""
    mock_overview = {
        "USDJPY": {
            "H1": {"count": 1000},
            "H4": {"count": 250}
        },
        "EURUSD": {
            "H1": {"count": 950}
        }
    }
    
    with patch('app.api.api_v1.endpoints.backtest.HistoricalDataService') as mock_service:
        mock_instance = mock_service.return_value
        mock_instance.get_data_overview.return_value = mock_overview
        mock_instance.supported_timeframes = {"H1": 60, "H4": 240, "D1": 1440}
        
        response = client.get("/api/v1/backtest/system/status")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["system_status"] == "operational"
        assert "data_status" in data
        assert "capabilities" in data
        assert "last_update" in data
        
        # データ統計の確認
        assert data["data_status"]["total_symbols"] == 2
        assert data["data_status"]["total_records"] == 2200  # 1000+250+950
        
        # 機能確認
        capabilities = data["capabilities"]
        assert capabilities["historical_data_fetch"] is True
        assert capabilities["backtest_execution"] is True
        assert capabilities["performance_analysis"] is True
        assert capabilities["report_generation"] is True


def test_invalid_request_validation(client):
    """無効リクエスト検証テスト"""
    # 無効なバックテストリクエスト
    invalid_request = {
        "symbol": "INVALID",
        "timeframe": "H1",
        "start_date": "invalid_date",
        "end_date": datetime.now().isoformat(),
        "initial_balance": -1000,  # 負の値
        "risk_per_trade": 1.5  # 100%超
    }
    
    response = client.post(
        "/api/v1/backtest/backtest/run",
        json=invalid_request
    )
    
    assert response.status_code == 422  # Validation Error


def test_error_handling_in_endpoints(client, sample_backtest_request):
    """エンドポイントエラーハンドリングテスト"""
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine:
        mock_instance = mock_engine.return_value
        mock_instance.run_backtest.side_effect = Exception("内部エラー")
        
        response = client.post(
            "/api/v1/backtest/backtest/run",
            json=sample_backtest_request
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "バックテスト実行エラー" in data["detail"]


def test_background_task_handling(client, sample_historical_data_request):
    """バックグラウンドタスク処理テスト"""
    # バックグラウンドタスクは実際には実行されないが、エンドポイントが正常に応答することを確認
    response = client.post(
        "/api/v1/backtest/historical-data/fetch",
        json=sample_historical_data_request
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "started"
    # バックグラウンドタスクが正常に追加されたことを確認


def test_response_models_structure(client, sample_backtest_request):
    """レスポンスモデル構造テスト"""
    mock_metrics = BacktestMetrics(
        total_trades=10,
        winning_trades=7,
        losing_trades=3,
        win_rate=0.7,
        total_profit=1000.0,
        profit_factor=2.33,
        max_drawdown=0.05
    )
    
    mock_result = BacktestResult(
        config=BacktestConfig(symbol="USDJPY", timeframe="H1"),
        status=BacktestStatus.COMPLETED,
        metrics=mock_metrics,
        positions=[],
        equity_curve=pd.DataFrame(),
        daily_returns=pd.DataFrame(),
        execution_time=1.0
    )
    
    with patch('app.api.api_v1.endpoints.backtest.BacktestEngine') as mock_engine:
        mock_instance = mock_engine.return_value
        mock_instance.run_backtest.return_value = mock_result
        
        response = client.post(
            "/api/v1/backtest/backtest/run",
            json=sample_backtest_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # レスポンス構造の確認
        required_fields = ["backtest_id", "status", "config", "metrics", "execution_time"]
        for field in required_fields:
            assert field in data
        
        # メトリクス構造の確認
        metrics_fields = ["total_trades", "win_rate", "total_profit", "profit_factor", "max_drawdown"]
        for field in metrics_fields:
            assert field in data["metrics"]