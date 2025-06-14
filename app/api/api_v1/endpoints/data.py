from fastapi import APIRouter, Query, Depends
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional, List
from app.core.database import get_db
from app.services.data_service import DataService
from app.schemas.price_data import PriceDataResponse
from app.schemas.mt5 import DataFetchRequest

router = APIRouter()


@router.get("/price/{symbol}", response_model=List[PriceDataResponse])
async def get_price_data(
    symbol: str,
    timeframe: str = Query("M1", description="時間足 (M1, M5, M15, M30, H1, H4, D1)"),
    limit: int = Query(100, ge=1, le=1000, description="取得データ数"),
    db: Session = Depends(get_db)
):
    data_service = DataService(db)
    price_data = data_service.get_price_data(symbol, timeframe, limit)
    return price_data


@router.post("/fetch/{symbol}")
async def fetch_price_data(
    symbol: str,
    timeframe: str = Query("M1", description="時間足"),
    count: int = Query(100, ge=1, le=5000, description="取得データ数"),
    db: Session = Depends(get_db)
):
    data_service = DataService(db)
    saved_count = data_service.fetch_and_save_data(symbol, timeframe, count)
    
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "saved_count": saved_count,
        "message": f"{saved_count}件のデータを保存しました",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/analysis/{symbol}")
async def get_analysis_data(
    symbol: str,
    timeframe: str = Query("H4", description="時間足"),
    db: Session = Depends(get_db)
):
    from app.services.analysis_engine import AnalysisEngineService
    
    analysis_engine = AnalysisEngineService(db)
    result = analysis_engine.analyze_symbol(symbol, timeframe)
    
    return result


@router.get("/analysis/{symbol}/history")
async def get_analysis_history(
    symbol: str,
    limit: int = Query(10, ge=1, le=50, description="取得件数"),
    db: Session = Depends(get_db)
):
    from app.services.analysis_engine import AnalysisEngineService
    
    analysis_engine = AnalysisEngineService(db)
    history = analysis_engine.get_analysis_history(symbol, limit)
    
    return {
        "symbol": symbol,
        "history": history,
        "count": len(history),
        "timestamp": datetime.now().isoformat()
    }