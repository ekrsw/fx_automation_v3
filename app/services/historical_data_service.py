import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, asc
import logging
from dataclasses import dataclass
from enum import Enum

from app.services.mt5_service import MT5Service
from app.models.price_data import PriceData
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataQualityStatus(str, Enum):
    """データ品質ステータス"""
    GOOD = "good"
    HAS_GAPS = "has_gaps"
    HAS_DUPLICATES = "has_duplicates"
    INVALID_FORMAT = "invalid_format"
    MISSING_DATA = "missing_data"


@dataclass
class DataQualityReport:
    """データ品質レポート"""
    symbol: str
    timeframe: str
    total_records: int
    date_range: Tuple[datetime, datetime]
    gaps_count: int
    duplicates_count: int
    invalid_records: int
    quality_score: float  # 0-100
    status: DataQualityStatus
    issues: List[str]
    recommendations: List[str]


@dataclass
class HistoricalDataSummary:
    """履歴データサマリー"""
    symbol: str
    timeframe: str
    records_fetched: int
    records_saved: int
    date_range: Tuple[datetime, datetime]
    quality_report: DataQualityReport
    processing_time: float


class HistoricalDataService:
    """履歴データ取得・管理サービス"""
    
    def __init__(self, db: Session):
        self.db = db
        self.mt5_service = MT5Service()
        
        # サポートする通貨ペアとタイムフレーム
        self.supported_symbols = [
            "USDJPY", "EURUSD", "GBPUSD", "AUDUSD", "USDCAD",
            "USDCHF", "NZDUSD", "EURJPY", "GBPJPY", "AUDJPY"
        ]
        
        self.supported_timeframes = {
            "M1": "1分足",
            "M5": "5分足", 
            "M15": "15分足",
            "M30": "30分足",
            "H1": "1時間足",
            "H4": "4時間足",
            "D1": "日足"
        }
        
        # データ品質基準
        self.quality_thresholds = {
            'max_gap_ratio': 0.05,      # 5%以下のギャップ
            'max_duplicate_ratio': 0.01, # 1%以下の重複
            'min_completeness': 0.95     # 95%以上の完全性
        }
    
    def fetch_and_store_historical_data(
        self,
        symbol: str = "USDJPY",
        timeframe: str = "H1",
        years_back: int = 5,
        force_update: bool = False
    ) -> HistoricalDataSummary:
        """履歴データを取得してデータベースに保存"""
        
        start_time = datetime.now()
        logger.info(f"履歴データ取得開始: {symbol} {timeframe} ({years_back}年間)")
        
        try:
            # MT5接続確認
            if not self.mt5_service.connect():
                raise Exception("MT5接続に失敗しました")
            
            # データ取得期間設定
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            # 既存データ確認
            if not force_update:
                existing_range = self._get_existing_data_range(symbol, timeframe)
                if existing_range:
                    start_date = max(start_date, existing_range[1])
                    logger.info(f"既存データを考慮して取得期間を調整: {start_date} から {end_date}")
            
            # MT5からデータ取得
            raw_data = self._fetch_mt5_data(symbol, timeframe, start_date, end_date)
            
            if raw_data is None or len(raw_data) == 0:
                logger.warning(f"データが取得できませんでした: {symbol} {timeframe}")
                return self._create_empty_summary(symbol, timeframe, start_time)
            
            # データ正規化
            normalized_data = self._normalize_data(raw_data, symbol, timeframe)
            
            # データ品質チェック
            quality_report = self._check_data_quality(normalized_data, symbol, timeframe)
            
            # データクリーニング
            cleaned_data = self._clean_data(normalized_data, quality_report)
            
            # データベース保存
            saved_count = self._save_to_database(cleaned_data, force_update)
            
            # 処理時間計算
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # サマリー作成
            summary = HistoricalDataSummary(
                symbol=symbol,
                timeframe=timeframe,
                records_fetched=len(raw_data),
                records_saved=saved_count,
                date_range=(normalized_data['datetime'].min(), normalized_data['datetime'].max()),
                quality_report=quality_report,
                processing_time=processing_time
            )
            
            logger.info(f"履歴データ取得完了: {saved_count}件保存, 処理時間: {processing_time:.2f}秒")
            return summary
            
        except Exception as e:
            logger.error(f"履歴データ取得エラー: {str(e)}")
            raise
        finally:
            self.mt5_service.disconnect()
    
    def fetch_all_symbols_data(
        self,
        timeframe: str = "H1",
        years_back: int = 3,
        force_update: bool = False
    ) -> List[HistoricalDataSummary]:
        """全通貨ペアの履歴データを取得"""
        
        summaries = []
        logger.info(f"全通貨ペアの履歴データ取得開始: {len(self.supported_symbols)}ペア")
        
        for symbol in self.supported_symbols:
            try:
                summary = self.fetch_and_store_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    years_back=years_back,
                    force_update=force_update
                )
                summaries.append(summary)
                logger.info(f"{symbol} 完了: {summary.records_saved}件保存")
                
            except Exception as e:
                logger.error(f"{symbol} の取得に失敗: {str(e)}")
                continue
        
        logger.info(f"全通貨ペア処理完了: {len(summaries)}ペア成功")
        return summaries
    
    def _fetch_mt5_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """MT5から生データを取得"""
        
        # 一度に取得する最大期間（MT5の制限を考慮）
        max_period_days = 365
        
        all_data = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_period_days), end_date)
            
            # MT5からデータ取得
            data = self.mt5_service.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=current_start
            )
            
            if data is not None and len(data) > 0:
                # 期間フィルター
                filtered_data = data[
                    (data['datetime'] >= current_start) & 
                    (data['datetime'] <= current_end)
                ]
                all_data.append(filtered_data)
                logger.debug(f"{symbol} {current_start.date()} - {current_end.date()}: {len(filtered_data)}件取得")
            
            current_start = current_end + timedelta(seconds=1)
        
        if not all_data:
            return None
        
        # 全データを結合
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 重複除去と並び替え
        combined_data = combined_data.drop_duplicates(subset=['datetime']).sort_values('datetime')
        combined_data = combined_data.reset_index(drop=True)
        
        logger.info(f"MT5データ取得完了: {symbol} {len(combined_data)}件")
        return combined_data
    
    def _normalize_data(self, data: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """データ正規化"""
        
        normalized = data.copy()
        
        # カラム名統一
        column_mapping = {
            'time': 'datetime',
            'tick_volume': 'volume'
        }
        normalized = normalized.rename(columns=column_mapping)
        
        # 必要カラムの確認
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in normalized.columns:
                logger.warning(f"必須カラム {col} が不足")
                if col == 'volume':
                    normalized[col] = 0  # ボリュームが無い場合は0で補完
                else:
                    raise ValueError(f"必須カラム {col} が不足しています")
        
        # データ型変換
        normalized['datetime'] = pd.to_datetime(normalized['datetime'])
        for col in ['open', 'high', 'low', 'close']:
            normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        normalized['volume'] = pd.to_numeric(normalized['volume'], errors='coerce').fillna(0)
        
        # シンボルとタイムフレーム追加
        normalized['symbol'] = symbol
        normalized['timeframe'] = timeframe
        
        # 価格の妥当性チェック
        normalized = self._validate_price_data(normalized)
        
        return normalized[['symbol', 'timeframe', 'datetime', 'open', 'high', 'low', 'close', 'volume']]
    
    def _validate_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """価格データの妥当性チェック"""
        
        validated = data.copy()
        
        # NaN除去
        validated = validated.dropna(subset=['open', 'high', 'low', 'close'])
        
        # 価格の論理チェック (high >= low, high >= open, high >= close, low <= open, low <= close)
        price_logic_mask = (
            (validated['high'] >= validated['low']) &
            (validated['high'] >= validated['open']) &
            (validated['high'] >= validated['close']) &
            (validated['low'] <= validated['open']) &
            (validated['low'] <= validated['close'])
        )
        
        invalid_count = (~price_logic_mask).sum()
        if invalid_count > 0:
            logger.warning(f"価格論理エラーのデータを除外: {invalid_count}件")
            validated = validated[price_logic_mask]
        
        # 異常値検出（価格変動が10%を超える場合）
        validated['price_change'] = validated['close'].pct_change().abs()
        outlier_mask = validated['price_change'] > 0.1
        outlier_count = outlier_mask.sum()
        
        if outlier_count > 0:
            logger.warning(f"異常値を検出: {outlier_count}件（10%超の価格変動）")
            # 異常値は除外せず、ログのみ出力
        
        validated = validated.drop('price_change', axis=1)
        return validated
    
    def _check_data_quality(self, data: pd.DataFrame, symbol: str, timeframe: str) -> DataQualityReport:
        """データ品質チェック"""
        
        if len(data) == 0:
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_records=0,
                date_range=(datetime.now(), datetime.now()),
                gaps_count=0,
                duplicates_count=0,
                invalid_records=0,
                quality_score=0.0,
                status=DataQualityStatus.MISSING_DATA,
                issues=["データが存在しません"],
                recommendations=["データ取得期間を確認してください"]
            )
        
        # 基本統計
        total_records = len(data)
        date_range = (data['datetime'].min(), data['datetime'].max())
        
        # 重複チェック
        duplicates_count = data['datetime'].duplicated().sum()
        
        # ギャップチェック
        gaps_count = self._count_time_gaps(data, timeframe)
        
        # 無効レコード数
        invalid_records = data.isnull().any(axis=1).sum()
        
        # 品質スコア計算
        completeness = (total_records - invalid_records) / total_records if total_records > 0 else 0
        gap_ratio = gaps_count / total_records if total_records > 0 else 0
        duplicate_ratio = duplicates_count / total_records if total_records > 0 else 0
        
        quality_score = 100 * (
            completeness * 0.5 +  # 完全性50%
            (1 - gap_ratio) * 0.3 +  # ギャップ30%
            (1 - duplicate_ratio) * 0.2  # 重複20%
        )
        
        # ステータス判定
        issues = []
        recommendations = []
        
        if gap_ratio > self.quality_thresholds['max_gap_ratio']:
            issues.append(f"データギャップが多すぎます（{gap_ratio:.2%}）")
            recommendations.append("より細かい期間でのデータ取得を検討してください")
        
        if duplicate_ratio > self.quality_thresholds['max_duplicate_ratio']:
            issues.append(f"重複データが多すぎます（{duplicate_ratio:.2%}）")
            recommendations.append("データクリーニング処理を強化してください")
        
        if completeness < self.quality_thresholds['min_completeness']:
            issues.append(f"データ完全性が低すぎます（{completeness:.2%}）")
            recommendations.append("データソースの品質を確認してください")
        
        # ステータス決定
        if quality_score >= 90:
            status = DataQualityStatus.GOOD
        elif gaps_count > 0:
            status = DataQualityStatus.HAS_GAPS
        elif duplicates_count > 0:
            status = DataQualityStatus.HAS_DUPLICATES
        elif invalid_records > 0:
            status = DataQualityStatus.INVALID_FORMAT
        else:
            status = DataQualityStatus.GOOD
        
        return DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_records=total_records,
            date_range=date_range,
            gaps_count=gaps_count,
            duplicates_count=duplicates_count,
            invalid_records=invalid_records,
            quality_score=quality_score,
            status=status,
            issues=issues,
            recommendations=recommendations
        )
    
    def _count_time_gaps(self, data: pd.DataFrame, timeframe: str) -> int:
        """時間ギャップをカウント"""
        
        if len(data) < 2:
            return 0
        
        # タイムフレームに応じた期待間隔
        timeframe_minutes = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440
        }
        
        expected_interval = timedelta(minutes=timeframe_minutes.get(timeframe, 60))
        
        # 時間差計算
        time_diffs = data['datetime'].diff()[1:]  # 最初の要素をスキップ
        
        # 期待間隔の1.5倍を超える場合をギャップとみなす
        gap_threshold = expected_interval * 1.5
        gaps = time_diffs > gap_threshold
        
        return gaps.sum()
    
    def _clean_data(self, data: pd.DataFrame, quality_report: DataQualityReport) -> pd.DataFrame:
        """データクリーニング"""
        
        cleaned = data.copy()
        
        # 重複除去
        if quality_report.duplicates_count > 0:
            cleaned = cleaned.drop_duplicates(subset=['datetime'])
            logger.info(f"重複データを除去: {quality_report.duplicates_count}件")
        
        # 無効データ除去
        cleaned = cleaned.dropna(subset=['open', 'high', 'low', 'close'])
        
        # 時間順ソート
        cleaned = cleaned.sort_values('datetime').reset_index(drop=True)
        
        return cleaned
    
    def _save_to_database(self, data: pd.DataFrame, force_update: bool = False) -> int:
        """データベースへの保存"""
        
        if len(data) == 0:
            return 0
        
        saved_count = 0
        
        try:
            for _, row in data.iterrows():
                # 既存データチェック
                existing = self.db.query(PriceData).filter(
                    and_(
                        PriceData.symbol == row['symbol'],
                        PriceData.timeframe == row['timeframe'],
                        PriceData.datetime == row['datetime']
                    )
                ).first()
                
                if existing and not force_update:
                    continue  # 既存データをスキップ
                
                if existing and force_update:
                    # 既存データを更新
                    existing.open = row['open']
                    existing.high = row['high']
                    existing.low = row['low']
                    existing.close = row['close']
                    existing.volume = row['volume']
                else:
                    # 新規データを追加
                    price_data = PriceData(
                        symbol=row['symbol'],
                        timeframe=row['timeframe'],
                        datetime=row['datetime'],
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    self.db.add(price_data)
                
                saved_count += 1
                
                # バッチコミット（1000件ごと）
                if saved_count % 1000 == 0:
                    self.db.commit()
                    logger.debug(f"中間コミット: {saved_count}件")
            
            # 最終コミット
            self.db.commit()
            logger.info(f"データベース保存完了: {saved_count}件")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"データベース保存エラー: {str(e)}")
            raise
        
        return saved_count
    
    def _get_existing_data_range(self, symbol: str, timeframe: str) -> Optional[Tuple[datetime, datetime]]:
        """既存データの日付範囲を取得"""
        
        try:
            min_date = self.db.query(PriceData.datetime).filter(
                and_(
                    PriceData.symbol == symbol,
                    PriceData.timeframe == timeframe
                )
            ).order_by(asc(PriceData.datetime)).first()
            
            max_date = self.db.query(PriceData.datetime).filter(
                and_(
                    PriceData.symbol == symbol,
                    PriceData.timeframe == timeframe
                )
            ).order_by(desc(PriceData.datetime)).first()
            
            if min_date and max_date:
                return (min_date[0], max_date[0])
            
        except Exception as e:
            logger.error(f"既存データ範囲取得エラー: {str(e)}")
        
        return None
    
    def _create_empty_summary(self, symbol: str, timeframe: str, start_time: datetime) -> HistoricalDataSummary:
        """空のサマリーを作成"""
        
        quality_report = DataQualityReport(
            symbol=symbol,
            timeframe=timeframe,
            total_records=0,
            date_range=(datetime.now(), datetime.now()),
            gaps_count=0,
            duplicates_count=0,
            invalid_records=0,
            quality_score=0.0,
            status=DataQualityStatus.MISSING_DATA,
            issues=["データが取得できませんでした"],
            recommendations=["MT5接続とシンボル設定を確認してください"]
        )
        
        return HistoricalDataSummary(
            symbol=symbol,
            timeframe=timeframe,
            records_fetched=0,
            records_saved=0,
            date_range=(datetime.now(), datetime.now()),
            quality_report=quality_report,
            processing_time=(datetime.now() - start_time).total_seconds()
        )
    
    def get_data_overview(self) -> Dict[str, Any]:
        """データベース内のデータ概要を取得"""
        
        try:
            # 通貨ペア別・タイムフレーム別統計
            overview = {}
            
            for symbol in self.supported_symbols:
                symbol_data = {}
                
                for timeframe in self.supported_timeframes.keys():
                    count = self.db.query(PriceData).filter(
                        and_(
                            PriceData.symbol == symbol,
                            PriceData.timeframe == timeframe
                        )
                    ).count()
                    
                    if count > 0:
                        date_range = self._get_existing_data_range(symbol, timeframe)
                        symbol_data[timeframe] = {
                            'count': count,
                            'date_range': date_range
                        }
                
                if symbol_data:
                    overview[symbol] = symbol_data
            
            return overview
            
        except Exception as e:
            logger.error(f"データ概要取得エラー: {str(e)}")
            return {}
    
    def validate_historical_data(self, symbol: str, timeframe: str) -> DataQualityReport:
        """保存済み履歴データの検証"""
        
        try:
            # データベースからデータ取得
            db_data = self.db.query(PriceData).filter(
                and_(
                    PriceData.symbol == symbol,
                    PriceData.timeframe == timeframe
                )
            ).order_by(PriceData.datetime).all()
            
            if not db_data:
                return DataQualityReport(
                    symbol=symbol,
                    timeframe=timeframe,
                    total_records=0,
                    date_range=(datetime.now(), datetime.now()),
                    gaps_count=0,
                    duplicates_count=0,
                    invalid_records=0,
                    quality_score=0.0,
                    status=DataQualityStatus.MISSING_DATA,
                    issues=["データが存在しません"],
                    recommendations=["履歴データを取得してください"]
                )
            
            # DataFrameに変換
            data = pd.DataFrame([
                {
                    'symbol': d.symbol,
                    'timeframe': d.timeframe,
                    'datetime': d.datetime,
                    'open': d.open,
                    'high': d.high,
                    'low': d.low,
                    'close': d.close,
                    'volume': d.volume
                }
                for d in db_data
            ])
            
            # 品質チェック実行
            return self._check_data_quality(data, symbol, timeframe)
            
        except Exception as e:
            logger.error(f"データ検証エラー: {str(e)}")
            return DataQualityReport(
                symbol=symbol,
                timeframe=timeframe,
                total_records=0,
                date_range=(datetime.now(), datetime.now()),
                gaps_count=0,
                duplicates_count=0,
                invalid_records=0,
                quality_score=0.0,
                status=DataQualityStatus.INVALID_FORMAT,
                issues=[f"検証エラー: {str(e)}"],
                recommendations=["システム管理者に連絡してください"]
            )