from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging

from app.core.database import get_db
from app.services.historical_data_service import HistoricalDataService, HistoricalDataSummary
from app.services.backtest_engine import BacktestEngine, BacktestConfig, BacktestResult, BacktestStatus
from app.services.performance_analyzer import PerformanceAnalyzer, ComprehensiveAnalysis
from app.services.report_generator import ReportGenerator, ReportConfig, ReportType, ReportFormat
from pydantic import BaseModel, Field

router = APIRouter()
logger = logging.getLogger(__name__)


# Request Models
class HistoricalDataRequest(BaseModel):
    symbol: str = "USDJPY"
    timeframe: str = "H1"
    years_back: int = Field(default=3, ge=1, le=10)
    force_update: bool = False


class BacktestRequest(BaseModel):
    symbol: str = "USDJPY"
    timeframe: str = "H1"
    start_date: datetime
    end_date: datetime
    initial_balance: float = Field(default=100000.0, ge=1000.0)
    risk_per_trade: float = Field(default=0.02, ge=0.001, le=0.1)
    commission_per_lot: float = Field(default=5.0, ge=0.0)
    spread_pips: float = Field(default=2.0, ge=0.0)
    min_signal_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    min_risk_reward: float = Field(default=1.5, ge=0.1)
    max_positions: int = Field(default=3, ge=1, le=10)


class OptimizationRequest(BaseModel):
    base_config: BacktestRequest
    parameter_ranges: Dict[str, List[float]]


class ReportRequest(BaseModel):
    report_type: str = "single_strategy"  # single_strategy, strategy_comparison, optimization_report
    format: str = "html"  # html, json, csv
    title: str = "バックテストレポート"
    include_charts: bool = True
    include_detailed_trades: bool = False


# Response Models
class DataQualityResponse(BaseModel):
    symbol: str
    timeframe: str
    total_records: int
    quality_score: float
    status: str
    issues: List[str]
    recommendations: List[str]


class BacktestResponse(BaseModel):
    backtest_id: str
    status: str
    config: Dict[str, Any]
    metrics: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


# Services
def get_historical_data_service(db: Session = Depends(get_db)) -> HistoricalDataService:
    return HistoricalDataService(db)


def get_backtest_engine(db: Session = Depends(get_db)) -> BacktestEngine:
    return BacktestEngine(db)


def get_performance_analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer()


def get_report_generator() -> ReportGenerator:
    return ReportGenerator()


# Historical Data Endpoints
@router.post("/historical-data/fetch")
async def fetch_historical_data(
    request: HistoricalDataRequest,
    background_tasks: BackgroundTasks,
    historical_service: HistoricalDataService = Depends(get_historical_data_service)
):
    """履歴データ取得"""
    try:
        # バックグラウンドでデータ取得開始
        background_tasks.add_task(
            historical_service.fetch_and_store_historical_data,
            symbol=request.symbol,
            timeframe=request.timeframe,
            years_back=request.years_back,
            force_update=request.force_update
        )
        
        return {
            "status": "started",
            "message": f"{request.symbol} {request.timeframe}の履歴データ取得を開始しました",
            "symbol": request.symbol,
            "timeframe": request.timeframe,
            "years_back": request.years_back
        }
        
    except Exception as e:
        logger.error(f"履歴データ取得開始エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"履歴データ取得開始エラー: {str(e)}"
        )


@router.post("/historical-data/fetch-all")
async def fetch_all_historical_data(
    timeframe: str = "H1",
    years_back: int = 3,
    force_update: bool = False,
    background_tasks: BackgroundTasks = None,
    historical_service: HistoricalDataService = Depends(get_historical_data_service)
):
    """全通貨ペアの履歴データ取得"""
    try:
        # バックグラウンドで全通貨ペアデータ取得開始
        background_tasks.add_task(
            historical_service.fetch_all_symbols_data,
            timeframe=timeframe,
            years_back=years_back,
            force_update=force_update
        )
        
        return {
            "status": "started",
            "message": f"全通貨ペア {timeframe}の履歴データ取得を開始しました",
            "timeframe": timeframe,
            "years_back": years_back,
            "symbols_count": len(historical_service.supported_symbols)
        }
        
    except Exception as e:
        logger.error(f"全履歴データ取得開始エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"全履歴データ取得開始エラー: {str(e)}"
        )


@router.get("/historical-data/overview")
async def get_data_overview(
    historical_service: HistoricalDataService = Depends(get_historical_data_service)
):
    """データベース内の履歴データ概要"""
    try:
        overview = historical_service.get_data_overview()
        
        return {
            "overview": overview,
            "summary": {
                "total_symbols": len(overview),
                "available_timeframes": list(historical_service.supported_timeframes.keys()),
                "supported_symbols": historical_service.supported_symbols
            }
        }
        
    except Exception as e:
        logger.error(f"データ概要取得エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"データ概要取得エラー: {str(e)}"
        )


@router.get("/historical-data/quality/{symbol}/{timeframe}")
async def validate_data_quality(
    symbol: str,
    timeframe: str,
    historical_service: HistoricalDataService = Depends(get_historical_data_service)
) -> DataQualityResponse:
    """データ品質検証"""
    try:
        quality_report = historical_service.validate_historical_data(symbol, timeframe)
        
        return DataQualityResponse(
            symbol=quality_report.symbol,
            timeframe=quality_report.timeframe,
            total_records=quality_report.total_records,
            quality_score=quality_report.quality_score,
            status=quality_report.status.value,
            issues=quality_report.issues,
            recommendations=quality_report.recommendations
        )
        
    except Exception as e:
        logger.error(f"データ品質検証エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"データ品質検証エラー: {str(e)}"
        )


# Backtest Endpoints
@router.post("/backtest/run")
async def run_backtest(
    request: BacktestRequest,
    backtest_engine: BacktestEngine = Depends(get_backtest_engine)
) -> BacktestResponse:
    """バックテスト実行"""
    try:
        # 設定作成
        config = BacktestConfig(
            symbol=request.symbol,
            timeframe=request.timeframe,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_balance=request.initial_balance,
            risk_per_trade=request.risk_per_trade,
            commission_per_lot=request.commission_per_lot,
            spread_pips=request.spread_pips,
            min_signal_confidence=request.min_signal_confidence,
            min_risk_reward=request.min_risk_reward,
            max_positions=request.max_positions
        )
        
        # バックテスト実行
        result = backtest_engine.run_backtest(config)
        
        # レスポンス作成
        backtest_id = f"bt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        response_data = {
            "backtest_id": backtest_id,
            "status": result.status.value,
            "config": {
                "symbol": config.symbol,
                "timeframe": config.timeframe,
                "start_date": config.start_date.isoformat(),
                "end_date": config.end_date.isoformat(),
                "initial_balance": config.initial_balance,
                "risk_per_trade": config.risk_per_trade
            },
            "execution_time": result.execution_time
        }
        
        if result.status == BacktestStatus.COMPLETED:
            response_data["metrics"] = {
                "total_trades": result.metrics.total_trades,
                "win_rate": result.metrics.win_rate,
                "total_profit": result.metrics.total_profit,
                "profit_factor": result.metrics.profit_factor,
                "max_drawdown": result.metrics.max_drawdown,
                "average_win": result.metrics.average_win,
                "average_loss": result.metrics.average_loss,
                "largest_win": result.metrics.largest_win,
                "largest_loss": result.metrics.largest_loss
            }
        else:
            response_data["error_message"] = result.error_message
        
        return BacktestResponse(**response_data)
        
    except Exception as e:
        logger.error(f"バックテスト実行エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"バックテスト実行エラー: {str(e)}"
        )


@router.post("/backtest/optimize")
async def run_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    backtest_engine: BacktestEngine = Depends(get_backtest_engine)
):
    """パラメータ最適化"""
    try:
        # 基本設定作成
        base_config = BacktestConfig(
            symbol=request.base_config.symbol,
            timeframe=request.base_config.timeframe,
            start_date=request.base_config.start_date,
            end_date=request.base_config.end_date,
            initial_balance=request.base_config.initial_balance,
            risk_per_trade=request.base_config.risk_per_trade,
            commission_per_lot=request.base_config.commission_per_lot,
            spread_pips=request.base_config.spread_pips,
            min_signal_confidence=request.base_config.min_signal_confidence,
            min_risk_reward=request.base_config.min_risk_reward,
            max_positions=request.base_config.max_positions
        )
        
        # バックグラウンドで最適化実行
        background_tasks.add_task(
            backtest_engine.run_optimization,
            base_config=base_config,
            parameter_ranges=request.parameter_ranges
        )
        
        optimization_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return {
            "optimization_id": optimization_id,
            "status": "started",
            "message": "パラメータ最適化を開始しました",
            "parameter_ranges": request.parameter_ranges,
            "estimated_combinations": len(list(request.parameter_ranges.values())[0]) if request.parameter_ranges else 0
        }
        
    except Exception as e:
        logger.error(f"最適化開始エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"最適化開始エラー: {str(e)}"
        )


# Analysis Endpoints
@router.post("/analysis/performance")
async def analyze_performance(
    backtest_request: BacktestRequest,
    backtest_engine: BacktestEngine = Depends(get_backtest_engine),
    analyzer: PerformanceAnalyzer = Depends(get_performance_analyzer)
):
    """パフォーマンス分析"""
    try:
        # バックテスト実行
        config = BacktestConfig(
            symbol=backtest_request.symbol,
            timeframe=backtest_request.timeframe,
            start_date=backtest_request.start_date,
            end_date=backtest_request.end_date,
            initial_balance=backtest_request.initial_balance,
            risk_per_trade=backtest_request.risk_per_trade,
            commission_per_lot=backtest_request.commission_per_lot,
            spread_pips=backtest_request.spread_pips,
            min_signal_confidence=backtest_request.min_signal_confidence,
            min_risk_reward=backtest_request.min_risk_reward,
            max_positions=backtest_request.max_positions
        )
        
        backtest_result = backtest_engine.run_backtest(config)
        
        if backtest_result.status != BacktestStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"バックテスト失敗: {backtest_result.error_message}"
            )
        
        # パフォーマンス分析
        analysis = analyzer.analyze_performance(backtest_result)
        
        return {
            "analysis_id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "performance_score": {
                "overall_score": analysis.performance_score.overall_score,
                "rating": analysis.performance_score.rating.value,
                "profitability_score": analysis.performance_score.profitability_score,
                "consistency_score": analysis.performance_score.consistency_score,
                "risk_management_score": analysis.performance_score.risk_management_score,
                "efficiency_score": analysis.performance_score.efficiency_score
            },
            "executive_summary": analysis.executive_summary,
            "detailed_analysis": analysis.detailed_analysis,
            "strengths": analysis.performance_score.strengths,
            "weaknesses": analysis.performance_score.weaknesses,
            "recommendations": analysis.performance_score.recommendations
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"パフォーマンス分析エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"パフォーマンス分析エラー: {str(e)}"
        )


# Report Endpoints
@router.post("/reports/generate")
async def generate_report(
    backtest_request: BacktestRequest,
    report_request: ReportRequest,
    backtest_engine: BacktestEngine = Depends(get_backtest_engine),
    report_generator: ReportGenerator = Depends(get_report_generator)
):
    """レポート生成"""
    try:
        # バックテスト実行
        config = BacktestConfig(
            symbol=backtest_request.symbol,
            timeframe=backtest_request.timeframe,
            start_date=backtest_request.start_date,
            end_date=backtest_request.end_date,
            initial_balance=backtest_request.initial_balance,
            risk_per_trade=backtest_request.risk_per_trade,
            commission_per_lot=backtest_request.commission_per_lot,
            spread_pips=backtest_request.spread_pips,
            min_signal_confidence=backtest_request.min_signal_confidence,
            min_risk_reward=backtest_request.min_risk_reward,
            max_positions=backtest_request.max_positions
        )
        
        backtest_result = backtest_engine.run_backtest(config)
        
        if backtest_result.status != BacktestStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"バックテスト失敗: {backtest_result.error_message}"
            )
        
        # レポート設定
        report_config = ReportConfig(
            report_type=ReportType(report_request.report_type),
            format=ReportFormat(report_request.format),
            title=report_request.title,
            include_charts=report_request.include_charts,
            include_detailed_trades=report_request.include_detailed_trades
        )
        
        # レポート生成
        report = report_generator.generate_single_strategy_report(
            backtest_result, report_config
        )
        
        return {
            "report_id": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "file_path": report.file_path,
            "file_size": report.file_size,
            "generation_time": report.generation_time.isoformat(),
            "format": report.config.format.value,
            "download_url": f"/api/v1/reports/download/{report.file_path.split('/')[-1]}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"レポート生成エラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"レポート生成エラー: {str(e)}"
        )


@router.get("/reports/download/{filename}")
async def download_report(filename: str):
    """レポートダウンロード"""
    try:
        from fastapi.responses import FileResponse
        import os
        
        file_path = f"reports/{filename}"
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="レポートファイルが見つかりません"
            )
        
        return FileResponse(
            path=file_path,
            filename=filename,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"レポートダウンロードエラー: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"レポートダウンロードエラー: {str(e)}"
        )


# Utility Endpoints
@router.get("/backtest/presets")
async def get_backtest_presets():
    """バックテストプリセット設定"""
    return {
        "timeframes": ["M1", "M5", "M15", "M30", "H1", "H4", "D1"],
        "symbols": ["USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "AUDJPY"],
        "preset_configs": {
            "conservative": {
                "risk_per_trade": 0.01,
                "min_signal_confidence": 0.8,
                "min_risk_reward": 2.0,
                "max_positions": 2
            },
            "moderate": {
                "risk_per_trade": 0.02,
                "min_signal_confidence": 0.7,
                "min_risk_reward": 1.5,
                "max_positions": 3
            },
            "aggressive": {
                "risk_per_trade": 0.05,
                "min_signal_confidence": 0.6,
                "min_risk_reward": 1.2,
                "max_positions": 5
            }
        },
        "optimization_ranges": {
            "risk_per_trade": [0.01, 0.02, 0.03, 0.04, 0.05],
            "min_signal_confidence": [0.6, 0.65, 0.7, 0.75, 0.8],
            "min_risk_reward": [1.0, 1.2, 1.5, 2.0, 2.5],
            "max_positions": [1, 2, 3, 4, 5]
        }
    }


@router.get("/system/status")
async def get_system_status(
    historical_service: HistoricalDataService = Depends(get_historical_data_service)
):
    """システム状態取得"""
    try:
        data_overview = historical_service.get_data_overview()
        
        # 利用可能なデータ統計
        total_symbols = len(data_overview)
        total_records = 0
        
        for symbol_data in data_overview.values():
            for timeframe_data in symbol_data.values():
                total_records += timeframe_data['count']
        
        return {
            "system_status": "operational",
            "data_status": {
                "total_symbols": total_symbols,
                "total_records": total_records,
                "available_symbols": list(data_overview.keys()),
                "supported_timeframes": list(historical_service.supported_timeframes.keys())
            },
            "capabilities": {
                "historical_data_fetch": True,
                "backtest_execution": True,
                "performance_analysis": True,
                "report_generation": True,
                "parameter_optimization": True
            },
            "last_update": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"システム状態取得エラー: {str(e)}")
        return {
            "system_status": "error",
            "error_message": str(e),
            "last_update": datetime.now().isoformat()
        }