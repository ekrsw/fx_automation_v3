from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.core.database import get_db
from app.services.trading_engine import TradingEngineService, ExecutionMode, OrderRequest, OrderType
from app.services.risk_manager import RiskManagerService
from app.services.signal_generator import TradingSignal, SignalType
from app.models.positions import Position, PositionType, PositionStatus
from pydantic import BaseModel

router = APIRouter()


# Request/Response Models
class OpenPositionRequest(BaseModel):
    symbol: str
    position_type: str  # "buy" or "sell"
    lot_size: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None


class ClosePositionRequest(BaseModel):
    position_id: int
    reason: Optional[str] = "Manual close"


class ExecuteModeRequest(BaseModel):
    mode: str  # "live", "simulation", "paper"


class ExecuteSignalRequest(BaseModel):
    signal_type: str  # "buy", "sell"
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confidence: float = 0.8
    reasoning: Optional[str] = None


# 取引エンジンインスタンス取得
def get_trading_engine(db: Session = Depends(get_db)) -> TradingEngineService:
    return TradingEngineService(db=db, execution_mode=ExecutionMode.SIMULATION)


def get_risk_manager(db: Session = Depends(get_db)) -> RiskManagerService:
    return RiskManagerService(db=db)


@router.get("/status")
async def get_trading_status(
    trading_engine: TradingEngineService = Depends(get_trading_engine)
):
    """取引状態取得"""
    try:
        summary = trading_engine.get_trading_summary()
        monitoring = trading_engine.monitor_positions()
        
        return {
            "trading_enabled": True,
            "execution_mode": trading_engine.execution_mode.value,
            "last_update": datetime.now().isoformat(),
            "summary": summary,
            "portfolio": monitoring['summary'],
            "warnings": monitoring.get('warnings', []),
            "message": "取引システム稼働中"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"状態取得エラー: {str(e)}"
        )


@router.post("/start")
async def start_trading():
    """取引開始"""
    return {
        "status": "success",
        "message": "自動取引機能は今後のアップデートで実装予定です",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/stop")
async def stop_trading():
    """取引停止"""
    return {
        "status": "success", 
        "message": "自動取引機能は今後のアップデートで実装予定です",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/positions/open")
async def open_position(
    request: OpenPositionRequest,
    trading_engine: TradingEngineService = Depends(get_trading_engine)
):
    """手動ポジションオープン"""
    try:
        # ポジションタイプ変換
        position_type = PositionType.BUY if request.position_type.lower() == "buy" else PositionType.SELL
        
        # 注文リクエスト作成
        order_request = OrderRequest(
            symbol=request.symbol,
            order_type=OrderType.MARKET,
            position_type=position_type,
            lot_size=request.lot_size,
            price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            comment=request.comment or "Manual order",
            strategy_name="ManualTrading"
        )
        
        # 注文実行
        result = trading_engine._execute_order(order_request)
        
        if result.success:
            return {
                "status": "success",
                "position_id": result.position_id,
                "ticket": result.ticket,
                "execution_price": result.execution_price,
                "message": result.message,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.message
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ポジションオープンエラー: {str(e)}"
        )


@router.post("/positions/close")
async def close_position(
    request: ClosePositionRequest,
    trading_engine: TradingEngineService = Depends(get_trading_engine)
):
    """手動ポジションクローズ"""
    try:
        result = trading_engine.close_position(
            position_id=request.position_id,
            reason=request.reason
        )
        
        if result.success:
            return {
                "status": "success",
                "message": result.message,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.message
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ポジションクローズエラー: {str(e)}"
        )


@router.get("/positions")
async def get_positions(
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """ポジション一覧取得"""
    try:
        query = db.query(Position)
        
        if status_filter:
            if status_filter == "open":
                query = query.filter(Position.status == PositionStatus.OPEN)
            elif status_filter == "closed":
                query = query.filter(Position.status == PositionStatus.CLOSED)
            elif status_filter == "pending":
                query = query.filter(Position.status == PositionStatus.PENDING)
        
        positions = query.order_by(Position.created_at.desc()).limit(50).all()
        
        return {
            "positions": [
                {
                    "id": pos.id,
                    "symbol": pos.symbol,
                    "position_type": pos.position_type.value,
                    "status": pos.status.value,
                    "lot_size": pos.lot_size,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "exit_price": pos.exit_price,
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "profit_loss": pos.profit_loss,
                    "created_at": pos.created_at.isoformat() if pos.created_at else None,
                    "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                    "closed_at": pos.closed_at.isoformat() if pos.closed_at else None,
                    "comments": pos.comments
                }
                for pos in positions
            ],
            "total_count": len(positions)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"ポジション取得エラー: {str(e)}"
        )


@router.post("/positions/close-all")
async def close_all_positions(
    trading_engine: TradingEngineService = Depends(get_trading_engine),
    db: Session = Depends(get_db)
):
    """緊急時全ポジションクローズ"""
    try:
        open_positions = db.query(Position).filter(
            Position.status == PositionStatus.OPEN
        ).all()
        
        results = []
        success_count = 0
        
        for position in open_positions:
            result = trading_engine.close_position(
                position_id=position.id,
                reason="Emergency close all"
            )
            
            results.append({
                "position_id": position.id,
                "success": result.success,
                "message": result.message
            })
            
            if result.success:
                success_count += 1
        
        return {
            "status": "completed",
            "total_positions": len(open_positions),
            "closed_successfully": success_count,
            "failed_closes": len(open_positions) - success_count,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"全ポジションクローズエラー: {str(e)}"
        )


@router.post("/signals/execute")
async def execute_signal(
    request: ExecuteSignalRequest,
    trading_engine: TradingEngineService = Depends(get_trading_engine)
):
    """シグナル手動実行"""
    try:
        # シグナルタイプ変換
        signal_type = SignalType.BUY if request.signal_type.lower() == "buy" else SignalType.SELL
        
        # TradingSignal作成
        signal = TradingSignal(
            signal_type=signal_type,
            entry_price=request.entry_price,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit,
            confidence=request.confidence,
            reasoning=request.reasoning or "Manual signal execution",
            timestamp=datetime.now()
        )
        
        # シグナル実行
        result = trading_engine.execute_signal(signal)
        
        if result.success:
            return {
                "status": "success",
                "position_id": result.position_id,
                "ticket": result.ticket,
                "execution_price": result.execution_price,
                "message": result.message,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.message
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"シグナル実行エラー: {str(e)}"
        )


@router.get("/risk/summary")
async def get_risk_summary(
    risk_manager: RiskManagerService = Depends(get_risk_manager)
):
    """リスク管理サマリー"""
    try:
        summary = risk_manager.get_risk_summary()
        
        return {
            "risk_summary": summary,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"リスクサマリー取得エラー: {str(e)}"
        )


@router.post("/execution-mode")
async def set_execution_mode(
    request: ExecuteModeRequest,
    trading_engine: TradingEngineService = Depends(get_trading_engine)
):
    """実行モード変更"""
    try:
        mode_mapping = {
            "live": ExecutionMode.LIVE,
            "simulation": ExecutionMode.SIMULATION,
            "paper": ExecutionMode.PAPER
        }
        
        if request.mode not in mode_mapping:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"無効な実行モード: {request.mode}"
            )
        
        new_mode = mode_mapping[request.mode]
        trading_engine.set_execution_mode(new_mode)
        
        return {
            "status": "success",
            "previous_mode": trading_engine.execution_mode.value,
            "new_mode": new_mode.value,
            "message": f"実行モードを{new_mode.value}に変更しました",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"実行モード変更エラー: {str(e)}"
        )


@router.get("/monitoring")
async def get_monitoring_data(
    trading_engine: TradingEngineService = Depends(get_trading_engine)
):
    """取引監視データ取得"""
    try:
        monitoring_result = trading_engine.monitor_positions()
        
        return {
            "monitoring": monitoring_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"監視データ取得エラー: {str(e)}"
        )