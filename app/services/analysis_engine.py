import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session

from app.services.dow_theory import DowTheoryService, TrendAnalysis
from app.services.elliott_wave import ElliottWaveService, WavePattern
from app.services.signal_generator import SignalGeneratorService, TradingSignal
from app.services.technical_indicators import TechnicalIndicatorFactory
from app.services.data_service import DataService
from app.models.technical_analysis import TechnicalAnalysis
import json


class AnalysisEngineService:
    """統合分析エンジンサービス - 柔軟性と拡張性を重視した設計"""
    
    def __init__(self, db: Session):
        self.db = db
        self.data_service = DataService(db)
        
        # 各分析サービスを初期化
        self.dow_service = DowTheoryService()
        self.elliott_service = ElliottWaveService()
        self.signal_service = SignalGeneratorService()
        
        # 設定可能なパラメータ
        self.config = {
            'zigzag_deviation': 5.0,
            'min_data_points': 50,
            'analysis_timeframes': ['M1', 'M5', 'M15', 'H1', 'H4', 'D1'],
            'save_results': True
        }
    
    def analyze_symbol(self, symbol: str, timeframe: str = 'H4') -> Dict[str, Any]:
        """指定シンボルの包括的分析を実行"""
        try:
            # データ取得
            price_data = self.data_service.get_price_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=200  # 十分な履歴データ
            )
            
            if len(price_data) < self.config['min_data_points']:
                return self._create_error_result(
                    f"データ不足: {len(price_data)}件 (最小{self.config['min_data_points']}件必要)"
                )
            
            # DataFrame変換
            df = self._convert_to_dataframe(price_data)
            
            # 各分析を実行
            analysis_results = self._run_comprehensive_analysis(df)
            
            # 結果の統合
            integrated_result = self._integrate_analysis_results(
                symbol, timeframe, analysis_results
            )
            
            # 結果保存 (オプション)
            if self.config['save_results']:
                self._save_analysis_results(symbol, timeframe, integrated_result)
            
            return integrated_result
            
        except Exception as e:
            return self._create_error_result(f"分析エラー: {str(e)}")
    
    def _convert_to_dataframe(self, price_data: List) -> pd.DataFrame:
        """価格データをDataFrameに変換"""
        data = []
        for pd_item in price_data:
            data.append({
                'datetime': pd_item.datetime,
                'open': float(pd_item.open),
                'high': float(pd_item.high),
                'low': float(pd_item.low),
                'close': float(pd_item.close),
                'volume': pd_item.volume
            })
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)
        
        return df
    
    def _run_comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """包括的分析を実行"""
        results = {}
        
        try:
            # ダウ理論分析
            dow_analysis = self.dow_service.analyze(df)
            results['dow_theory'] = {
                'trend_direction': dow_analysis.trend_direction.value,
                'confidence': dow_analysis.confidence,
                'swing_points_count': len(dow_analysis.swing_points),
                'higher_highs': len(dow_analysis.higher_highs),
                'higher_lows': len(dow_analysis.higher_lows),
                'lower_highs': len(dow_analysis.lower_highs),
                'lower_lows': len(dow_analysis.lower_lows),
                'raw_analysis': dow_analysis
            }
        except Exception as e:
            results['dow_theory'] = {'error': str(e)}
        
        try:
            # エリオット波動分析
            elliott_pattern = self.elliott_service.analyze(df)
            results['elliott_wave'] = {
                'pattern_type': elliott_pattern.pattern_type.value,
                'waves_count': len(elliott_pattern.waves),
                'completion_percentage': elliott_pattern.completion_percentage,
                'current_wave': elliott_pattern.waves[-1].wave_label.value if elliott_pattern.waves else None,
                'next_target': {
                    'price': elliott_pattern.next_target.price,
                    'ratio': elliott_pattern.next_target.ratio,
                    'type': elliott_pattern.next_target.level_type
                } if elliott_pattern.next_target else None,
                'invalidation_level': elliott_pattern.invalidation_level,
                'raw_pattern': elliott_pattern
            }
        except Exception as e:
            results['elliott_wave'] = {'error': str(e)}
        
        try:
            # テクニカル指標
            rsi = TechnicalIndicatorFactory.create('rsi', period=14)
            atr = TechnicalIndicatorFactory.create('atr', period=14)
            
            rsi_values = rsi.calculate(df)
            atr_values = atr.calculate(df)
            
            results['technical_indicators'] = {
                'rsi_current': float(rsi_values.iloc[-1]) if len(rsi_values) > 0 else None,
                'rsi_overbought': float(rsi_values.iloc[-1]) > 70 if len(rsi_values) > 0 else False,
                'rsi_oversold': float(rsi_values.iloc[-1]) < 30 if len(rsi_values) > 0 else False,
                'atr_current': float(atr_values.iloc[-1]) if len(atr_values) > 0 else None,
                'volatility_percentile': self._calculate_volatility_percentile(atr_values) if len(atr_values) > 0 else None
            }
        except Exception as e:
            results['technical_indicators'] = {'error': str(e)}
        
        try:
            # シグナル生成
            signal = self.signal_service.generate_signal(df)
            results['trading_signal'] = {
                'signal_type': signal.signal_type.value,
                'strength': signal.strength.value,
                'confidence': signal.confidence,
                'entry_price': signal.entry_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'reasoning': signal.reasoning,
                'raw_signal': signal
            }
        except Exception as e:
            results['trading_signal'] = {'error': str(e)}
        
        return results
    
    def _integrate_analysis_results(self, symbol: str, timeframe: str, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """分析結果を統合"""
        # 総合スコア計算
        overall_score = self._calculate_overall_score(analysis_results)
        
        # マーケット状況評価
        market_condition = self._evaluate_market_condition(analysis_results)
        
        # 推奨アクション
        recommended_action = self._determine_recommended_action(analysis_results)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'analysis_timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'market_condition': market_condition,
            'recommended_action': recommended_action,
            'detailed_analysis': analysis_results,
            'system_health': self._check_system_health(analysis_results)
        }
    
    def _calculate_overall_score(self, results: Dict[str, Any]) -> Dict[str, float]:
        """総合スコアを計算 (0-100)"""
        scores = {}
        
        # ダウ理論スコア
        if 'dow_theory' in results and 'confidence' in results['dow_theory']:
            scores['dow_score'] = results['dow_theory']['confidence'] * 100
        
        # エリオット波動スコア
        if 'elliott_wave' in results and 'completion_percentage' in results['elliott_wave']:
            completion = results['elliott_wave']['completion_percentage']
            # 完了度が高いほどスコアが高い
            scores['elliott_score'] = min(completion, 100)
        
        # シグナルスコア
        if 'trading_signal' in results and 'confidence' in results['trading_signal']:
            scores['signal_score'] = results['trading_signal']['confidence'] * 100
        
        # 総合スコア
        if scores:
            scores['overall'] = sum(scores.values()) / len(scores)
        else:
            scores['overall'] = 0
        
        return scores
    
    def _evaluate_market_condition(self, results: Dict[str, Any]) -> str:
        """マーケット状況を評価"""
        conditions = []
        
        # ダウ理論から
        if 'dow_theory' in results:
            trend = results['dow_theory'].get('trend_direction', 'unknown')
            confidence = results['dow_theory'].get('confidence', 0)
            
            if confidence > 0.7:
                conditions.append(f"明確な{trend}トレンド")
            elif confidence > 0.4:
                conditions.append(f"弱い{trend}トレンド")
            else:
                conditions.append("トレンドレス")
        
        # テクニカル指標から
        if 'technical_indicators' in results:
            rsi = results['technical_indicators'].get('rsi_current')
            if rsi:
                if rsi > 70:
                    conditions.append("過買い状態")
                elif rsi < 30:
                    conditions.append("過売り状態")
                else:
                    conditions.append("中立状態")
        
        # エリオット波動から
        if 'elliott_wave' in results:
            wave = results['elliott_wave'].get('current_wave')
            if wave:
                conditions.append(f"エリオット{wave}波")
        
        return " | ".join(conditions) if conditions else "分析不可"
    
    def _determine_recommended_action(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """推奨アクションを決定"""
        signal_result = results.get('trading_signal', {})
        
        if 'error' in signal_result:
            return {
                'action': 'wait',
                'reason': 'シグナル分析エラー',
                'priority': 'low'
            }
        
        signal_type = signal_result.get('signal_type', 'hold')
        strength = signal_result.get('strength', 'weak')
        confidence = signal_result.get('confidence', 0)
        
        if signal_type in ['buy', 'sell'] and confidence > 0.6:
            priority = 'high' if strength in ['strong', 'very_strong'] else 'medium'
            return {
                'action': signal_type,
                'reason': f"{strength}の{signal_type}シグナル (信頼度: {confidence:.2f})",
                'priority': priority,
                'entry_price': signal_result.get('entry_price'),
                'stop_loss': signal_result.get('stop_loss'),
                'take_profit': signal_result.get('take_profit'),
                'risk_reward': signal_result.get('risk_reward_ratio')
            }
        
        return {
            'action': 'wait',
            'reason': '条件未満のシグナル',
            'priority': 'low'
        }
    
    def _calculate_volatility_percentile(self, atr_values: pd.Series, 
                                       lookback_period: int = 50) -> Optional[float]:
        """ボラティリティパーセンタイルを計算"""
        if len(atr_values) < lookback_period:
            return None
        
        recent_atr = atr_values.iloc[-lookback_period:]
        current_atr = atr_values.iloc[-1]
        
        percentile = (recent_atr < current_atr).sum() / len(recent_atr) * 100
        return float(percentile)
    
    def _check_system_health(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """システム健全性をチェック"""
        errors = []
        warnings = []
        
        for module, result in results.items():
            if isinstance(result, dict) and 'error' in result:
                errors.append(f"{module}: {result['error']}")
        
        # データ品質チェック
        if 'dow_theory' in results:
            swing_count = results['dow_theory'].get('swing_points_count', 0)
            if swing_count < 5:
                warnings.append(f"スイングポイント不足: {swing_count}個")
        
        return {
            'status': 'error' if errors else ('warning' if warnings else 'healthy'),
            'errors': errors,
            'warnings': warnings
        }
    
    def _save_analysis_results(self, symbol: str, timeframe: str, 
                             results: Dict[str, Any]) -> None:
        """分析結果をデータベースに保存"""
        try:
            # シンプルな形式でJSONとして保存
            serializable_results = self._make_serializable(results)
            
            analysis = TechnicalAnalysis(
                symbol=symbol,
                datetime=datetime.now(),
                dow_trend=results['detailed_analysis']['dow_theory'].get('trend_direction', 'unknown'),
                elliott_wave_count=results['detailed_analysis']['elliott_wave'].get('current_wave', 'unknown'),
                swing_points=json.dumps(serializable_results.get('swing_points', [])),
                signals=json.dumps(serializable_results.get('signals', {}))
            )
            
            self.db.add(analysis)
            self.db.commit()
            
        except Exception as e:
            self.db.rollback()
            # ログに記録 (実装時は適切なロガーを使用)
            print(f"分析結果保存エラー: {str(e)}")
    
    def _make_serializable(self, obj: Any) -> Any:
        """オブジェクトをシリアライズ可能な形式に変換"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """エラー結果を作成"""
        return {
            'error': True,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """設定を更新"""
        self.config.update(new_config)
    
    def get_config(self) -> Dict[str, Any]:
        """現在の設定を取得"""
        return self.config.copy()
    
    def get_analysis_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """分析履歴を取得"""
        history = self.db.query(TechnicalAnalysis).filter(
            TechnicalAnalysis.symbol == symbol
        ).order_by(TechnicalAnalysis.datetime.desc()).limit(limit).all()
        
        return [{
            'datetime': h.datetime.isoformat(),
            'dow_trend': h.dow_trend,
            'elliott_wave': h.elliott_wave_count,
            'created_at': h.created_at.isoformat()
        } for h in history]