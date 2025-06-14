from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "FX Trading System"
    }


@router.get("/mt5")
async def mt5_health():
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            return {
                "status": "unhealthy",
                "message": "MT5初期化に失敗しました",
                "error": mt5.last_error()
            }
        
        terminal_info = mt5.terminal_info()
        mt5.shutdown()
        
        return {
            "status": "healthy",
            "mt5_version": terminal_info.build if terminal_info else "不明",
            "message": "MT5接続正常"
        }
    except ImportError:
        return {
            "status": "unhealthy",
            "message": "MetaTrader5パッケージがインストールされていません"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"MT5接続エラー: {str(e)}"
        }