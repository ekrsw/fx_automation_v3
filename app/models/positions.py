from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, Enum as SQLEnum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum

Base = declarative_base()


class PositionType(str, Enum):
    """ポジションタイプ"""
    BUY = "buy"
    SELL = "sell"


class PositionStatus(str, Enum):
    """ポジション状態"""
    PENDING = "pending"      # 待機中
    OPEN = "open"           # オープン
    CLOSED = "closed"       # クローズ済み
    CANCELLED = "cancelled" # キャンセル済み


class Position(Base):
    """ポジション管理テーブル"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, index=True)
    
    # 基本情報
    symbol = Column(String(10), nullable=False, index=True)
    position_type = Column(SQLEnum(PositionType), nullable=False)
    status = Column(SQLEnum(PositionStatus), default=PositionStatus.PENDING, index=True)
    
    # ポジションサイズと価格
    lot_size = Column(Float, nullable=False)  # ロットサイズ
    entry_price = Column(Float, nullable=True)  # エントリー価格
    current_price = Column(Float, nullable=True)  # 現在価格
    exit_price = Column(Float, nullable=True)  # エグジット価格
    
    # リスク管理
    stop_loss = Column(Float, nullable=True)    # ストップロス
    take_profit = Column(Float, nullable=True)  # テイクプロフィット
    risk_amount = Column(Float, nullable=True)  # リスク金額
    
    # 取引結果
    profit_loss = Column(Float, default=0.0)      # 損益
    commission = Column(Float, default=0.0)       # 手数料
    swap = Column(Float, default=0.0)             # スワップ
    net_profit = Column(Float, default=0.0)       # 純利益
    
    # MT5情報
    mt5_ticket = Column(Integer, nullable=True, unique=True)  # MT5チケット番号
    mt5_order_id = Column(Integer, nullable=True)             # MT5オーダーID
    
    # 分析情報
    strategy_name = Column(String(50), nullable=True)        # 戦略名
    signal_confidence = Column(Float, nullable=True)         # シグナル信頼度
    analysis_data = Column(Text, nullable=True)              # 分析データ（JSON）
    
    # タイムスタンプ
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    opened_at = Column(DateTime(timezone=True), nullable=True)
    closed_at = Column(DateTime(timezone=True), nullable=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 関連データ
    comments = Column(Text, nullable=True)  # コメント
    
    def __repr__(self):
        return f"<Position(id={self.id}, symbol={self.symbol}, type={self.position_type}, status={self.status}, lot={self.lot_size})>"
    
    @property
    def is_open(self) -> bool:
        """ポジションがオープン中か"""
        return self.status == PositionStatus.OPEN
    
    @property
    def is_closed(self) -> bool:
        """ポジションがクローズ済みか"""
        return self.status == PositionStatus.CLOSED
    
    @property
    def is_profitable(self) -> bool:
        """ポジションが利益を出しているか"""
        return self.net_profit > 0 if self.net_profit is not None else False
    
    @property
    def unrealized_pnl(self) -> float:
        """含み損益計算"""
        if not self.is_open or self.current_price is None or self.entry_price is None:
            return 0.0
        
        price_diff = self.current_price - self.entry_price
        if self.position_type == PositionType.SELL:
            price_diff = -price_diff
        
        # 簡易的な計算（実際はシンボルのピップ値と契約サイズを考慮）
        return price_diff * self.lot_size * 100000  # 仮の計算
    
    @property
    def risk_reward_ratio(self) -> float:
        """リスクリワード比計算"""
        if not all([self.entry_price, self.stop_loss, self.take_profit]):
            return 0.0
        
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.take_profit - self.entry_price)
        
        return reward / risk if risk > 0 else 0.0


class PositionHistory(Base):
    """ポジション履歴テーブル（パフォーマンス分析用）"""
    __tablename__ = "position_history"
    
    id = Column(Integer, primary_key=True, index=True)
    position_id = Column(Integer, nullable=False, index=True)
    
    # 履歴データ
    action = Column(String(20), nullable=False)  # open, modify, close, etc.
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    
    # 価格情報
    price = Column(Float, nullable=True)
    volume = Column(Float, nullable=True)
    
    # 変更内容
    old_values = Column(Text, nullable=True)  # 変更前の値（JSON）
    new_values = Column(Text, nullable=True)  # 変更後の値（JSON）
    
    # メタデータ
    reason = Column(String(100), nullable=True)  # 変更理由
    comments = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<PositionHistory(id={self.id}, position_id={self.position_id}, action={self.action})>"