import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from enum import Enum

from app.services.performance_analyzer import (
    PerformanceAnalyzer, ComprehensiveAnalysis, PerformanceScore, PerformanceRating,
    RiskMetrics, ReturnMetrics, RatioMetrics, TradingMetrics
)
from app.services.backtest_engine import (
    BacktestResult, BacktestConfig, BacktestMetrics, BacktestPosition, 
    BacktestStatus
)
from app.models.positions import PositionType

# ExitReasonクラスを定義
class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"
    TIMEOUT = "timeout"


@pytest.fixture
def performance_analyzer():
    """パフォーマンス分析器"""
    return PerformanceAnalyzer()


@pytest.fixture
def sample_backtest_config():
    """サンプルバックテスト設定"""
    return BacktestConfig(
        symbol="USDJPY",
        timeframe="H1",
        start_date=datetime(2023, 1, 1),
        end_date=datetime(2023, 12, 31),
        initial_balance=100000.0,
        risk_per_trade=0.02
    )


@pytest.fixture
def profitable_positions():
    """利益の出るポジション群"""
    positions = []
    base_time = datetime(2023, 1, 1)
    
    for i in range(10):
        position = BacktestPosition(
            id=i,
            symbol="USDJPY", 
            position_type="buy" if i % 2 == 0 else "sell",
            entry_time=base_time + timedelta(days=i*5),
            entry_price=150.0 + i * 0.1,
            lot_size=10000,
            exit_time=base_time + timedelta(days=i*5, hours=12),
            exit_price=150.0 + i * 0.1 + (0.5 if i % 3 != 0 else -0.2),  # 70%勝率
            net_profit=(50.0 if i % 3 != 0 else -20.0),  # 利益50, 損失-20
            profit_loss=(50.0 if i % 3 != 0 else -20.0),
            commission=5.0,
            exit_reason="take_profit" if i % 3 != 0 else "stop_loss"
        )
        positions.append(position)
    
    return positions


@pytest.fixture
def losing_positions():
    """損失の出るポジション群"""
    positions = []
    base_time = datetime(2023, 1, 1)
    
    for i in range(5):
        position = BacktestPosition(
            id=f"loss_pos_{i}",
            symbol="USDJPY",
            position_type="buy",
            entry_time=base_time + timedelta(days=i*10),
            exit_time=base_time + timedelta(days=i*10, hours=8),
            entry_price=150.0,
            exit_price=149.0,  # 全て損失
            lot_size=10000,
            net_profit=-100.0,
            profit_loss=-100.0,
            commission=5.0,
            exit_reason="stop_loss"
        )
        positions.append(position)
    
    return positions


@pytest.fixture
def sample_backtest_result(sample_backtest_config, profitable_positions):
    """サンプルバックテスト結果"""
    # メトリクス計算
    total_profit = sum(pos.net_profit for pos in profitable_positions)
    winning_trades = sum(1 for pos in profitable_positions if pos.net_profit > 0)
    losing_trades = len(profitable_positions) - winning_trades
    gross_profit = sum(pos.net_profit for pos in profitable_positions if pos.net_profit > 0)
    gross_loss = sum(pos.net_profit for pos in profitable_positions if pos.net_profit < 0)
    
    metrics = BacktestMetrics(
        total_trades=len(profitable_positions),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / len(profitable_positions),
        total_profit=total_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf'),
        average_win=gross_profit / winning_trades if winning_trades > 0 else 0,
        average_loss=gross_loss / losing_trades if losing_trades > 0 else 0,
        largest_win=max((pos.net_profit for pos in profitable_positions if pos.net_profit > 0), default=0),
        largest_loss=min((pos.net_profit for pos in profitable_positions if pos.net_profit < 0), default=0),
        max_drawdown=0.05,  # 5%のドローダウン
        max_consecutive_wins=3,
        max_consecutive_losses=1
    )
    
    # エクイティカーブ生成
    equity_data = []
    balance = sample_backtest_config.initial_balance
    
    for i, pos in enumerate(profitable_positions):
        balance += pos.net_profit
        equity_data.append({
            'datetime': pos.exit_time,
            'equity': balance,
            'drawdown': max(0, (max(eq['equity'] for eq in equity_data[:i+1]) - balance) / max(eq['equity'] for eq in equity_data[:i+1])) if equity_data else 0
        })
    
    equity_curve = pd.DataFrame(equity_data)
    
    # 月次リターン生成
    monthly_returns = pd.DataFrame([
        {'year': 2023, 'month': i+1, 'monthly_return': 0.02 + (i % 3) * 0.01, 'trades': 1}
        for i in range(12)
    ])
    
    # 日次リターン生成
    daily_returns = pd.DataFrame([
        {'date': datetime(2023, 1, 1) + timedelta(days=i), 'daily_return': 0.001 * (1 + i % 3)}
        for i in range(30)
    ])
    
    return BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=metrics,
        positions=profitable_positions,
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        execution_time=1.5,
        monthly_returns=monthly_returns
    )


def test_performance_analyzer_initialization(performance_analyzer):
    """パフォーマンス分析器初期化テスト"""
    assert performance_analyzer is not None
    assert hasattr(performance_analyzer, 'scoring_weights')
    assert hasattr(performance_analyzer, 'benchmarks')
    assert 'risk_free_rate' in performance_analyzer.benchmarks
    assert 'market_return' in performance_analyzer.benchmarks


def test_calculate_risk_metrics(performance_analyzer, sample_backtest_result):
    """リスクメトリクス計算テスト"""
    risk_metrics = performance_analyzer._calculate_risk_metrics(sample_backtest_result)
    
    assert isinstance(risk_metrics, RiskMetrics)
    assert risk_metrics.max_drawdown >= 0
    assert risk_metrics.volatility >= 0
    assert risk_metrics.value_at_risk_95 >= 0  # VaRは絶対値として計算される
    assert risk_metrics.conditional_var_95 >= 0
    assert risk_metrics.downside_deviation >= 0
    assert risk_metrics.beta is not None


def test_calculate_return_metrics(performance_analyzer, sample_backtest_result):
    """リターンメトリクス計算テスト"""
    return_metrics = performance_analyzer._calculate_return_metrics(sample_backtest_result)
    
    assert isinstance(return_metrics, ReturnMetrics)
    assert return_metrics.total_return is not None
    assert return_metrics.annualized_return is not None
    assert 0 <= return_metrics.monthly_win_rate <= 1
    assert return_metrics.best_month is not None
    assert return_metrics.worst_month is not None
    assert return_metrics.positive_months >= 0
    assert return_metrics.negative_months >= 0


def test_calculate_ratio_metrics(performance_analyzer, sample_backtest_result):
    """比率メトリクス計算テスト"""
    ratio_metrics = performance_analyzer._calculate_ratio_metrics(sample_backtest_result)
    
    assert isinstance(ratio_metrics, RatioMetrics)
    assert ratio_metrics.sharpe_ratio is not None
    assert ratio_metrics.sortino_ratio is not None
    assert ratio_metrics.calmar_ratio is not None
    assert ratio_metrics.treynor_ratio is not None
    assert ratio_metrics.information_ratio is not None


def test_calculate_trading_metrics(performance_analyzer, sample_backtest_result):
    """取引メトリクス計算テスト"""
    trading_metrics = performance_analyzer._calculate_trading_metrics(sample_backtest_result)
    
    assert isinstance(trading_metrics, TradingMetrics)
    assert trading_metrics.profit_factor > 0
    assert trading_metrics.recovery_factor is not None
    assert 0 <= trading_metrics.trade_efficiency <= 1
    assert trading_metrics.average_holding_period > 0
    assert trading_metrics.win_loss_ratio >= 0
    assert trading_metrics.expectancy is not None


def test_calculate_performance_score_excellent(performance_analyzer, sample_backtest_result):
    """パフォーマンススコア計算テスト - 優秀"""
    # 優秀な結果に調整
    sample_backtest_result.metrics.total_profit = 50000  # 50%利益
    sample_backtest_result.metrics.win_rate = 0.8  # 80%勝率
    sample_backtest_result.metrics.profit_factor = 3.0
    sample_backtest_result.metrics.max_drawdown = 0.02  # 2%ドローダウン
    
    performance_score = performance_analyzer._calculate_performance_score(sample_backtest_result)
    
    assert isinstance(performance_score, PerformanceScore)
    assert 0 <= performance_score.overall_score <= 100
    assert performance_score.rating in [rating.value for rating in PerformanceRating]
    assert performance_score.profitability_score > 0
    assert performance_score.consistency_score > 0
    assert performance_score.risk_management_score > 0
    assert performance_score.efficiency_score > 0
    assert len(performance_score.strengths) > 0
    assert isinstance(performance_score.recommendations, list)


def test_calculate_performance_score_poor(performance_analyzer, sample_backtest_result):
    """パフォーマンススコア計算テスト - 劣悪"""
    # 劣悪な結果に調整
    sample_backtest_result.metrics.total_profit = -10000  # 損失
    sample_backtest_result.metrics.win_rate = 0.3  # 30%勝率
    sample_backtest_result.metrics.profit_factor = 0.5
    sample_backtest_result.metrics.max_drawdown = 0.25  # 25%ドローダウン
    
    performance_score = performance_analyzer._calculate_performance_score(sample_backtest_result)
    
    assert isinstance(performance_score, PerformanceScore)
    assert performance_score.overall_score < 50
    assert performance_score.rating == PerformanceRating.POOR
    assert len(performance_score.weaknesses) > 0
    assert len(performance_score.recommendations) > 0


def test_generate_executive_summary(performance_analyzer, sample_backtest_result):
    """エグゼクティブサマリー生成テスト"""
    summary = performance_analyzer._generate_executive_summary(sample_backtest_result)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert "USDJPY" in summary  # 通貨ペア名が含まれる
    assert any(keyword in summary.lower() for keyword in ["利益", "勝率", "ドローダウン"])


def test_generate_detailed_analysis(performance_analyzer, sample_backtest_result):
    """詳細分析生成テスト"""
    analysis = performance_analyzer._generate_detailed_analysis(sample_backtest_result)
    
    assert isinstance(analysis, str)
    assert len(analysis) > 0
    assert "リスク分析" in analysis
    assert "リターン分析" in analysis
    assert "取引分析" in analysis


def test_analyze_performance_complete(performance_analyzer, sample_backtest_result):
    """完全なパフォーマンス分析テスト"""
    analysis = performance_analyzer.analyze_performance(sample_backtest_result)
    
    assert isinstance(analysis, ComprehensiveAnalysis)
    assert analysis.backtest_result == sample_backtest_result
    assert isinstance(analysis.risk_metrics, RiskMetrics)
    assert isinstance(analysis.return_metrics, ReturnMetrics)
    assert isinstance(analysis.ratio_metrics, RatioMetrics)
    assert isinstance(analysis.trading_metrics, TradingMetrics)
    assert isinstance(analysis.performance_score, PerformanceScore)
    assert isinstance(analysis.executive_summary, str)
    assert isinstance(analysis.detailed_analysis, str)


def test_compare_strategies(performance_analyzer, profitable_positions, losing_positions, sample_backtest_config):
    """戦略比較テスト"""
    # 2つの異なる結果を作成
    result1 = BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=BacktestMetrics(
            total_trades=len(profitable_positions),
            winning_trades=7,
            losing_trades=3,
            win_rate=0.7,
            total_profit=300.0,
            gross_profit=500.0,
            gross_loss=-200.0,
            profit_factor=2.5,
            max_drawdown=0.03
        ),
        positions=profitable_positions,
        equity_curve=pd.DataFrame([{'datetime': datetime.now(), 'balance': 100300}]),
        execution_time=1.0
    )
    
    result2 = BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=BacktestMetrics(
            total_trades=len(losing_positions),
            winning_trades=1,
            losing_trades=4,
            win_rate=0.2,
            total_profit=-400.0,
            gross_profit=100.0,
            gross_loss=-500.0,
            profit_factor=0.2,
            max_drawdown=0.15
        ),
        positions=losing_positions,
        equity_curve=pd.DataFrame([{'datetime': datetime.now(), 'balance': 99600}]),
        execution_time=0.8
    )
    
    comparison = performance_analyzer.compare_strategies([result1, result2])
    
    assert isinstance(comparison, dict)
    assert 'summary' in comparison
    assert 'strategy_comparison' in comparison
    assert 'rankings' in comparison
    assert 'best_overall' in comparison
    
    # 戦略1が戦略2より優秀であることを確認
    assert comparison['best_overall']['strategy_id'] == 'Strategy_1'


def test_calculate_volatility(performance_analyzer):
    """ボラティリティ計算テスト"""
    # サンプルリターンデータ
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03])
    
    volatility = performance_analyzer._calculate_volatility(returns)
    
    assert volatility > 0
    assert isinstance(volatility, float)


def test_calculate_var(performance_analyzer):
    """VaR計算テスト"""
    # サンプルリターンデータ
    returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, -0.05, 0.02, -0.01])
    
    var_95 = performance_analyzer._calculate_var(returns, confidence_level=0.95)
    var_99 = performance_analyzer._calculate_var(returns, confidence_level=0.99)
    
    assert var_95 <= 0  # VaRは負の値
    assert var_99 <= 0
    assert var_99 <= var_95  # 99%VaRは95%VaRより小さい（より極端）


def test_calculate_drawdown_series(performance_analyzer):
    """ドローダウン系列計算テスト"""
    # サンプルエクイティデータ
    equity_values = pd.Series([100000, 102000, 101000, 103000, 99000, 104000, 102000])
    
    drawdown_series = performance_analyzer._calculate_drawdown_series(equity_values)
    
    assert isinstance(drawdown_series, pd.Series)
    assert len(drawdown_series) == len(equity_values)
    assert all(dd <= 0 for dd in drawdown_series)  # ドローダウンは負またはゼロ
    assert drawdown_series.iloc[0] == 0  # 最初はドローダウンなし


def test_calculate_monthly_returns(performance_analyzer, sample_backtest_result):
    """月次リターン計算テスト"""
    monthly_returns = performance_analyzer._calculate_monthly_returns(sample_backtest_result)
    
    assert isinstance(monthly_returns, pd.DataFrame)
    assert 'year' in monthly_returns.columns
    assert 'month' in monthly_returns.columns
    assert 'return' in monthly_returns.columns
    assert len(monthly_returns) > 0


def test_edge_cases_empty_positions(performance_analyzer, sample_backtest_config):
    """エッジケース - 空ポジション"""
    empty_result = BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=BacktestMetrics(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_profit=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=0.0,
            max_drawdown=0.0
        ),
        positions=[],
        equity_curve=pd.DataFrame([{'datetime': datetime.now(), 'balance': 100000}]),
        execution_time=0.1
    )
    
    analysis = performance_analyzer.analyze_performance(empty_result)
    
    assert isinstance(analysis, ComprehensiveAnalysis)
    assert analysis.performance_score.overall_score == 0
    assert analysis.performance_score.rating == PerformanceRating.POOR


def test_edge_cases_single_trade(performance_analyzer, sample_backtest_config):
    """エッジケース - 単一取引"""
    single_position = [BacktestPosition(
        id="single",
        symbol="USDJPY",
        position_type="buy",
        entry_time=datetime.now() - timedelta(hours=1),
        exit_time=datetime.now(),
        entry_price=150.0,
        exit_price=150.5,
        lot_size=10000,
        net_profit=50.0,
        is_closed=True
    )]
    
    single_result = BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=BacktestMetrics(
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            win_rate=1.0,
            total_profit=50.0,
            gross_profit=50.0,
            gross_loss=0.0,
            profit_factor=float('inf'),
            max_drawdown=0.0
        ),
        positions=single_position,
        equity_curve=pd.DataFrame([
            {'datetime': datetime.now() - timedelta(hours=1), 'balance': 100000},
            {'datetime': datetime.now(), 'balance': 100050}
        ]),
        execution_time=0.1
    )
    
    analysis = performance_analyzer.analyze_performance(single_result)
    
    assert isinstance(analysis, ComprehensiveAnalysis)
    assert analysis.performance_score.overall_score > 0


def test_chart_generation_mock(performance_analyzer, sample_backtest_result):
    """チャート生成テスト（モック使用）"""
    with patch('matplotlib.pyplot.savefig') as mock_savefig:
        mock_savefig.return_value = None
        
        # チャート生成メソッドのテスト
        equity_chart = performance_analyzer._generate_equity_curve_chart(sample_backtest_result)
        assert isinstance(equity_chart, str)  # base64エンコードされた文字列
        
        monthly_chart = performance_analyzer._generate_monthly_returns_chart(sample_backtest_result)
        assert isinstance(monthly_chart, str)
        
        drawdown_chart = performance_analyzer._generate_drawdown_chart(sample_backtest_result)
        assert isinstance(drawdown_chart, str)


def test_performance_rating_classification(performance_analyzer):
    """パフォーマンス評価分類テスト"""
    # 各スコア範囲での評価テスト
    test_cases = [
        (95, PerformanceRating.EXCELLENT),
        (85, PerformanceRating.EXCELLENT),
        (75, PerformanceRating.GOOD),
        (65, PerformanceRating.GOOD),
        (55, PerformanceRating.AVERAGE),
        (45, PerformanceRating.AVERAGE),
        (35, PerformanceRating.POOR),
        (15, PerformanceRating.POOR)
    ]
    
    for score, expected_rating in test_cases:
        rating = performance_analyzer._classify_performance_rating(score)
        assert rating == expected_rating


def test_statistical_calculations(performance_analyzer):
    """統計計算テスト"""
    sample_data = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.04, -0.03, 0.02, -0.01, 0.05])
    
    # 基本統計
    mean_val = np.mean(sample_data)
    std_val = np.std(sample_data)
    
    assert mean_val is not None
    assert std_val > 0
    
    # Sharpe比類似計算
    sharpe_like = mean_val / std_val if std_val > 0 else 0
    assert isinstance(sharpe_like, float)


def test_comprehensive_analysis_attributes(performance_analyzer, sample_backtest_result):
    """包括的分析属性テスト"""
    analysis = performance_analyzer.analyze_performance(sample_backtest_result)
    
    # すべての必要な属性が存在することを確認
    required_attributes = [
        'backtest_result', 'risk_metrics', 'return_metrics', 'ratio_metrics',
        'trading_metrics', 'performance_score', 'executive_summary', 'detailed_analysis'
    ]
    
    for attr in required_attributes:
        assert hasattr(analysis, attr)
        assert getattr(analysis, attr) is not None