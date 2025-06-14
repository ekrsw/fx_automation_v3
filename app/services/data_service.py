from sqlalchemy.orm import Session
from app.models.price_data import PriceData
from app.services.mt5_service import MT5Service
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class DataService:
    def __init__(self, db: Session):
        self.db = db
        self.mt5_service = MT5Service()

    def save_price_data(self, symbol: str, timeframe: str, data_frame) -> int:
        saved_count = 0
        try:
            for _, row in data_frame.iterrows():
                # 既存データチェック
                existing = self.db.query(PriceData).filter(
                    PriceData.symbol == symbol,
                    PriceData.timeframe == timeframe,
                    PriceData.datetime == row['datetime']
                ).first()

                if not existing:
                    price_data = PriceData(
                        symbol=symbol,
                        timeframe=timeframe,
                        datetime=row['datetime'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    self.db.add(price_data)
                    saved_count += 1

            self.db.commit()
            logger.info(f"{symbol} {timeframe}: {saved_count}件のデータを保存")
            return saved_count

        except Exception as e:
            self.db.rollback()
            logger.error(f"データ保存エラー: {str(e)}")
            return 0

    def fetch_and_save_data(self, symbol: str, timeframe: str = "M1", count: int = 100) -> int:
        if not self.mt5_service.connect():
            logger.error("MT5接続失敗")
            return 0

        try:
            data = self.mt5_service.get_price_data(symbol, timeframe, count)
            if data is None or data.empty:
                logger.warning(f"データ取得失敗: {symbol} {timeframe}")
                return 0

            saved_count = self.save_price_data(symbol, timeframe, data)
            return saved_count

        finally:
            self.mt5_service.disconnect()

    def get_price_data(
        self,
        symbol: str,
        timeframe: str = "M1",
        limit: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PriceData]:
        query = self.db.query(PriceData).filter(
            PriceData.symbol == symbol,
            PriceData.timeframe == timeframe
        )

        if start_date:
            query = query.filter(PriceData.datetime >= start_date)
        if end_date:
            query = query.filter(PriceData.datetime <= end_date)

        return query.order_by(PriceData.datetime.desc()).limit(limit).all()

    def get_latest_price(self, symbol: str, timeframe: str = "M1") -> Optional[PriceData]:
        return self.db.query(PriceData).filter(
            PriceData.symbol == symbol,
            PriceData.timeframe == timeframe
        ).order_by(PriceData.datetime.desc()).first()