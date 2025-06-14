try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class OrderTypeAction(int, Enum):
    """MT5注文タイプ"""
    BUY = 0
    SELL = 1
    BUY_LIMIT = 2
    SELL_LIMIT = 3
    BUY_STOP = 4
    SELL_STOP = 5


@dataclass
class TradeRequest:
    """取引リクエスト"""
    action: int  # mt5.TRADE_ACTION_*
    symbol: str
    volume: float
    type: int  # OrderTypeAction
    price: float = 0.0
    sl: float = 0.0  # ストップロス
    tp: float = 0.0  # テイクプロフィット
    deviation: int = 20  # スリッページ
    magic: int = 12345  # マジックナンバー
    comment: str = "Auto trade"
    type_time: int = 0  # 有効期限タイプ
    expiration: int = 0  # 有効期限


@dataclass 
class TradeResult:
    """取引結果"""
    retcode: int
    deal: int
    order: int
    volume: float
    price: float
    bid: float
    ask: float
    comment: str
    request_id: int
    success: bool


class MT5Service:
    def __init__(self):
        self.connected = False

    def connect(self) -> bool:
        if mt5 is None:
            logger.error("MetaTrader5モジュールがインストールされていません")
            return False
            
        try:
            if not mt5.initialize():
                logger.error(f"MT5初期化失敗: {mt5.last_error()}")
                return False

            if settings.mt5_login and settings.mt5_password and settings.mt5_server:
                authorized = mt5.login(
                    login=settings.mt5_login,
                    password=settings.mt5_password,
                    server=settings.mt5_server
                )
                if not authorized:
                    logger.error(f"MT5ログイン失敗: {mt5.last_error()}")
                    return False

            self.connected = True
            logger.info("MT5接続成功")
            return True

        except Exception as e:
            logger.error(f"MT5接続エラー: {str(e)}")
            return False

    def disconnect(self):
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("MT5接続終了")

    def is_connected(self) -> bool:
        if mt5 is None:
            return False
        return self.connected and mt5.terminal_info() is not None

    def get_price_data(
        self,
        symbol: str,
        timeframe: str = "M1",
        count: int = 100,
        start_date: Optional[datetime] = None
    ) -> Optional[pd.DataFrame]:
        if not self.is_connected():
            logger.error("MT5未接続")
            return None

        try:
            # タイムフレーム変換
            tf_map = {
                "M1": mt5.TIMEFRAME_M1,
                "M5": mt5.TIMEFRAME_M5,
                "M15": mt5.TIMEFRAME_M15,
                "M30": mt5.TIMEFRAME_M30,
                "H1": mt5.TIMEFRAME_H1,
                "H4": mt5.TIMEFRAME_H4,
                "D1": mt5.TIMEFRAME_D1
            }

            if timeframe not in tf_map:
                logger.error(f"不正なタイムフレーム: {timeframe}")
                return None

            if start_date:
                rates = mt5.copy_rates_from(symbol, tf_map[timeframe], start_date, count)
            else:
                rates = mt5.copy_rates_range(
                    symbol, 
                    tf_map[timeframe], 
                    datetime.now() - timedelta(days=30), 
                    datetime.now()
                )

            if rates is None or len(rates) == 0:
                logger.warning(f"データ取得失敗: {symbol} {timeframe}")
                return None

            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.rename(columns={'time': 'datetime', 'tick_volume': 'volume'}, inplace=True)
            
            return df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"価格データ取得エラー: {str(e)}")
            return None

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        if not self.is_connected():
            return None

        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None

            return {
                "symbol": symbol,
                "bid": info.bid,
                "ask": info.ask,
                "spread": info.spread,
                "digits": info.digits,
                "point": info.point
            }
        except Exception as e:
            logger.error(f"シンボル情報取得エラー: {str(e)}")
            return None

    def get_account_info(self) -> Optional[dict]:
        if not self.is_connected():
            return None

        try:
            info = mt5.account_info()
            if info is None:
                return None

            return {
                "login": info.login,
                "balance": info.balance,
                "equity": info.equity,
                "profit": info.profit,
                "margin": info.margin,
                "currency": info.currency
            }
        except Exception as e:
            logger.error(f"アカウント情報取得エラー: {str(e)}")
            return None

    # === 取引機能 ===
    
    def send_order(self, trade_request: TradeRequest) -> TradeResult:
        """注文送信"""
        if not self.is_connected():
            return TradeResult(
                retcode=10004,  # NO_CONNECTION
                deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment="MT5未接続",
                request_id=0, success=False
            )
        
        if mt5 is None:
            return TradeResult(
                retcode=10005,  # NO_MT5
                deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment="MT5モジュール未インストール",
                request_id=0, success=False
            )
        
        try:
            # 取引リクエスト作成
            request = {
                "action": trade_request.action,
                "symbol": trade_request.symbol,
                "volume": trade_request.volume,
                "type": trade_request.type,
                "price": trade_request.price,
                "sl": trade_request.sl,
                "tp": trade_request.tp,
                "deviation": trade_request.deviation,
                "magic": trade_request.magic,
                "comment": trade_request.comment,
                "type_time": trade_request.type_time,
                "expiration": trade_request.expiration,
            }
            
            # 注文送信
            result = mt5.order_send(request)
            
            if result is None:
                return TradeResult(
                    retcode=10006,  # REQUEST_FAILED
                    deal=0, order=0, volume=0.0, price=0.0,
                    bid=0.0, ask=0.0, comment="注文送信失敗",
                    request_id=0, success=False
                )
            
            return TradeResult(
                retcode=result.retcode,
                deal=result.deal,
                order=result.order,
                volume=result.volume,
                price=result.price,
                bid=result.bid,
                ask=result.ask,
                comment=result.comment,
                request_id=result.request_id,
                success=result.retcode == mt5.TRADE_RETCODE_DONE
            )
            
        except Exception as e:
            logger.error(f"注文送信エラー: {str(e)}")
            return TradeResult(
                retcode=10007,  # EXCEPTION
                deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment=f"エラー: {str(e)}",
                request_id=0, success=False
            )
    
    def buy_market(self, symbol: str, volume: float, 
                   sl: float = 0.0, tp: float = 0.0, 
                   comment: str = "Buy market") -> TradeResult:
        """成行買い注文"""
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return TradeResult(
                retcode=10008, deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment="シンボル情報取得失敗",
                request_id=0, success=False
            )
        
        trade_request = TradeRequest(
            action=mt5.TRADE_ACTION_DEAL if mt5 else 1,
            symbol=symbol,
            volume=volume,
            type=OrderTypeAction.BUY,
            price=symbol_info["ask"],
            sl=sl,
            tp=tp,
            comment=comment
        )
        
        return self.send_order(trade_request)
    
    def sell_market(self, symbol: str, volume: float,
                    sl: float = 0.0, tp: float = 0.0,
                    comment: str = "Sell market") -> TradeResult:
        """成行売り注文"""
        symbol_info = self.get_symbol_info(symbol)
        if not symbol_info:
            return TradeResult(
                retcode=10008, deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment="シンボル情報取得失敗",
                request_id=0, success=False
            )
        
        trade_request = TradeRequest(
            action=mt5.TRADE_ACTION_DEAL if mt5 else 1,
            symbol=symbol,
            volume=volume,
            type=OrderTypeAction.SELL,
            price=symbol_info["bid"],
            sl=sl,
            tp=tp,
            comment=comment
        )
        
        return self.send_order(trade_request)
    
    def close_position(self, ticket: int) -> TradeResult:
        """ポジションクローズ"""
        if not self.is_connected() or mt5 is None:
            return TradeResult(
                retcode=10004, deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment="MT5未接続または未インストール",
                request_id=0, success=False
            )
        
        try:
            # ポジション情報取得
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return TradeResult(
                    retcode=10009, deal=0, order=0, volume=0.0, price=0.0,
                    bid=0.0, ask=0.0, comment="ポジションが見つかりません",
                    request_id=0, success=False
                )
            
            position = positions[0]
            
            # 反対売買のタイプ決定
            if position.type == mt5.ORDER_TYPE_BUY:
                trade_type = OrderTypeAction.SELL
                price = self.get_symbol_info(position.symbol)["bid"]
            else:
                trade_type = OrderTypeAction.BUY
                price = self.get_symbol_info(position.symbol)["ask"]
            
            trade_request = TradeRequest(
                action=mt5.TRADE_ACTION_DEAL,
                symbol=position.symbol,
                volume=position.volume,
                type=trade_type,
                price=price,
                comment=f"Close #{ticket}"
            )
            
            return self.send_order(trade_request)
            
        except Exception as e:
            logger.error(f"ポジションクローズエラー: {str(e)}")
            return TradeResult(
                retcode=10007, deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment=f"エラー: {str(e)}",
                request_id=0, success=False
            )
    
    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """ポジション一覧取得"""
        if not self.is_connected() or mt5 is None:
            return []
        
        try:
            positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
            if not positions:
                return []
            
            position_list = []
            for pos in positions:
                position_list.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": pos.type,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "comment": pos.comment,
                    "time": pos.time
                })
            
            return position_list
            
        except Exception as e:
            logger.error(f"ポジション取得エラー: {str(e)}")
            return []
    
    def get_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """注文一覧取得"""
        if not self.is_connected() or mt5 is None:
            return []
        
        try:
            orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
            if not orders:
                return []
            
            order_list = []
            for order in orders:
                order_list.append({
                    "ticket": order.ticket,
                    "symbol": order.symbol,
                    "type": order.type,
                    "volume_initial": order.volume_initial,
                    "volume_current": order.volume_current,
                    "price_open": order.price_open,
                    "sl": order.sl,
                    "tp": order.tp,
                    "comment": order.comment,
                    "time_setup": order.time_setup
                })
            
            return order_list
            
        except Exception as e:
            logger.error(f"注文取得エラー: {str(e)}")
            return []
    
    def modify_position(self, ticket: int, sl: float = None, tp: float = None) -> TradeResult:
        """ポジション修正（SL/TP変更）"""
        if not self.is_connected() or mt5 is None:
            return TradeResult(
                retcode=10004, deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment="MT5未接続または未インストール",
                request_id=0, success=False
            )
        
        try:
            # ポジション情報取得
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return TradeResult(
                    retcode=10009, deal=0, order=0, volume=0.0, price=0.0,
                    bid=0.0, ask=0.0, comment="ポジションが見つかりません",
                    request_id=0, success=False
                )
            
            position = positions[0]
            
            # 修正リクエスト作成
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "volume": position.volume,
                "position": ticket,
                "sl": sl if sl is not None else position.sl,
                "tp": tp if tp is not None else position.tp,
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                return TradeResult(
                    retcode=10006, deal=0, order=0, volume=0.0, price=0.0,
                    bid=0.0, ask=0.0, comment="修正リクエスト失敗",
                    request_id=0, success=False
                )
            
            return TradeResult(
                retcode=result.retcode,
                deal=result.deal,
                order=result.order,
                volume=result.volume,
                price=result.price,
                bid=result.bid,
                ask=result.ask,
                comment=result.comment,
                request_id=result.request_id,
                success=result.retcode == mt5.TRADE_RETCODE_DONE
            )
            
        except Exception as e:
            logger.error(f"ポジション修正エラー: {str(e)}")
            return TradeResult(
                retcode=10007, deal=0, order=0, volume=0.0, price=0.0,
                bid=0.0, ask=0.0, comment=f"エラー: {str(e)}",
                request_id=0, success=False
            )