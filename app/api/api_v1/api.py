from fastapi import APIRouter
from app.api.api_v1.endpoints import trading, data, health, backtest

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(trading.router, prefix="/trading", tags=["trading"])
api_router.include_router(data.router, prefix="/data", tags=["data"])
api_router.include_router(backtest.router, prefix="/backtest", tags=["backtest"])