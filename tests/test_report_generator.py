import pytest
import json
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import os
from enum import Enum

from app.services.report_generator import (
    ReportGenerator, ReportConfig, ReportFormat, ReportType, GeneratedReport
)
from app.services.backtest_engine import (
    BacktestResult, BacktestConfig, BacktestMetrics, BacktestPosition,
    BacktestStatus
)
from app.models.positions import PositionType
from app.services.performance_analyzer import (
    ComprehensiveAnalysis, PerformanceScore, PerformanceRating,
    RiskMetrics, ReturnMetrics, RatioMetrics, TradingMetrics
)

# ExitReasonクラスを定義
class ExitReason(str, Enum):
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MANUAL = "manual"
    TIMEOUT = "timeout"


@pytest.fixture
def temp_dir():
    """一時ディレクトリ"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def report_generator(temp_dir):
    """レポート生成器"""
    generator = ReportGenerator()
    # テンプレートディレクトリを一時ディレクトリに設定
    generator.template_dir = Path(temp_dir) / "templates"
    generator.template_dir.mkdir(exist_ok=True)
    generator._create_default_templates()
    return generator


@pytest.fixture
def sample_config():
    """サンプル設定"""
    return ReportConfig(
        report_type=ReportType.SINGLE_STRATEGY,
        format=ReportFormat.HTML,
        title="テストレポート",
        include_charts=True,
        include_detailed_trades=True
    )


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
def sample_positions():
    """サンプルポジション群"""
    positions = []
    base_time = datetime(2023, 1, 1)
    
    for i in range(5):
        position = BacktestPosition(
            id=f"pos_{i}",
            symbol="USDJPY",
            position_type=PositionType.BUY if i % 2 == 0 else PositionType.SELL,
            entry_time=base_time + timedelta(days=i*10),
            exit_time=base_time + timedelta(days=i*10, hours=12),
            entry_price=150.0 + i * 0.1,
            exit_price=150.0 + i * 0.1 + (0.5 if i % 2 == 0 else -0.3),
            lot_size=10000,
            stop_loss=149.0 + i * 0.1,
            take_profit=152.0 + i * 0.1,
            net_profit=(50.0 if i % 2 == 0 else -30.0),
            profit_loss=(50.0 if i % 2 == 0 else -30.0),
            commission=5.0,
            is_closed=True,
            exit_reason=ExitReason.TAKE_PROFIT if i % 2 == 0 else ExitReason.STOP_LOSS
        )
        positions.append(position)
    
    return positions


@pytest.fixture
def sample_backtest_result(sample_backtest_config, sample_positions):
    """サンプルバックテスト結果"""
    # メトリクス計算
    total_profit = sum(pos.net_profit for pos in sample_positions)
    winning_trades = sum(1 for pos in sample_positions if pos.net_profit > 0)
    losing_trades = len(sample_positions) - winning_trades
    gross_profit = sum(pos.net_profit for pos in sample_positions if pos.net_profit > 0)
    gross_loss = sum(pos.net_profit for pos in sample_positions if pos.net_profit < 0)
    
    metrics = BacktestMetrics(
        total_trades=len(sample_positions),
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=winning_trades / len(sample_positions),
        total_profit=total_profit,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=abs(gross_profit / gross_loss) if gross_loss != 0 else float('inf'),
        average_win=gross_profit / winning_trades if winning_trades > 0 else 0,
        average_loss=gross_loss / losing_trades if losing_trades > 0 else 0,
        largest_win=max((pos.net_profit for pos in sample_positions if pos.net_profit > 0), default=0),
        largest_loss=min((pos.net_profit for pos in sample_positions if pos.net_profit < 0), default=0),
        max_drawdown=0.03,
        consecutive_wins=2,
        consecutive_losses=1
    )
    
    # エクイティカーブ生成
    equity_data = []
    balance = sample_backtest_config.initial_balance
    
    for pos in sample_positions:
        balance += pos.net_profit
        equity_data.append({
            'datetime': pos.exit_time,
            'balance': balance
        })
    
    equity_curve = pd.DataFrame(equity_data)
    
    # 月次リターン生成
    monthly_returns = pd.DataFrame([
        {'year': 2023, 'month': i+1, 'return': 0.01 + (i % 3) * 0.005, 'trades': 1}
        for i in range(12)
    ])
    
    return BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=metrics,
        positions=sample_positions,
        equity_curve=equity_curve,
        monthly_returns=monthly_returns,
        execution_time=1.2,
        start_time=datetime.now() - timedelta(seconds=2),
        end_time=datetime.now()
    )


@pytest.fixture
def sample_comprehensive_analysis(sample_backtest_result):
    """サンプル包括的分析"""
    risk_metrics = RiskMetrics(
        max_drawdown=0.03,
        volatility=0.15,
        value_at_risk_95=-0.02,
        value_at_risk_99=-0.04,
        downside_deviation=0.08,
        beta=1.2
    )
    
    return_metrics = ReturnMetrics(
        total_return=0.05,
        annualized_return=0.06,
        monthly_win_rate=0.75,
        best_month=0.04,
        worst_month=-0.02,
        positive_months=9,
        negative_months=3
    )
    
    ratio_metrics = RatioMetrics(
        sharpe_ratio=1.8,
        sortino_ratio=2.1,
        calmar_ratio=2.0,
        treynor_ratio=0.05,
        information_ratio=0.8
    )
    
    trading_metrics = TradingMetrics(
        profit_factor=1.67,
        recovery_factor=1.5,
        trade_efficiency=0.7,
        average_holding_period=12.0,
        win_loss_ratio=1.67,
        expectancy=20.0
    )
    
    performance_score = PerformanceScore(
        overall_score=78.5,
        rating=PerformanceRating.GOOD,
        profitability_score=80.0,
        consistency_score=75.0,
        risk_management_score=82.0,
        efficiency_score=77.0,
        strengths=["高いプロフィットファクター", "良好なリスク管理"],
        weaknesses=["取引頻度が低い"],
        recommendations=["取引機会を増やす", "ストップロスを最適化"]
    )
    
    return ComprehensiveAnalysis(
        backtest_result=sample_backtest_result,
        risk_metrics=risk_metrics,
        return_metrics=return_metrics,
        ratio_metrics=ratio_metrics,
        trading_metrics=trading_metrics,
        performance_score=performance_score,
        executive_summary="テスト戦略は良好な成績を示している。",
        detailed_analysis="詳細な分析結果がここに表示される。",
        equity_curve_chart="mock_chart_data_1",
        monthly_returns_chart="mock_chart_data_2",
        drawdown_chart="mock_chart_data_3",
        trade_distribution_chart="mock_chart_data_4"
    )


def test_report_generator_initialization(report_generator):
    """レポート生成器初期化テスト"""
    assert report_generator.performance_analyzer is not None
    assert report_generator.template_dir.exists()
    assert (report_generator.template_dir / "single_strategy_report.html").exists()
    assert (report_generator.template_dir / "comparison_report.html").exists()
    assert (report_generator.template_dir / "optimization_report.html").exists()


def test_generate_html_report(report_generator, sample_backtest_result, sample_config, temp_dir):
    """HTMLレポート生成テスト"""
    sample_config.format = ReportFormat.HTML
    sample_config.output_path = f"{temp_dir}/test_report.html"
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analysis = Mock()
        mock_analysis.performance_score = Mock()
        mock_analysis.performance_score.overall_score = 78.5
        mock_analysis.performance_score.rating = PerformanceRating.GOOD
        mock_analysis.performance_score.strengths = ["高い利益率"]
        mock_analysis.performance_score.weaknesses = ["高いリスク"]
        mock_analysis.performance_score.recommendations = ["リスク軽減"]
        mock_analysis.risk_metrics = Mock()
        mock_analysis.risk_metrics.max_drawdown = 0.03
        mock_analysis.return_metrics = Mock()
        mock_analysis.ratio_metrics = Mock()
        mock_analysis.ratio_metrics.sharpe_ratio = 1.8
        mock_analysis.trading_metrics = Mock()
        mock_analysis.executive_summary = "テストサマリー"
        mock_analysis.detailed_analysis = "詳細分析"
        mock_analysis.backtest_result = sample_backtest_result
        mock_analysis.equity_curve_chart = None
        mock_analysis.monthly_returns_chart = None
        mock_analysis.drawdown_chart = None
        mock_analysis.trade_distribution_chart = None
        mock_analyze.return_value = mock_analysis
        
        report = report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)
        
        assert isinstance(report, GeneratedReport)
        assert report.config == sample_config
        assert len(report.content) > 0
        assert report.file_path == sample_config.output_path
        assert report.generation_time is not None
        assert report.file_size > 0
        
        # ファイルが実際に作成されているか確認
        assert os.path.exists(report.file_path)
        
        # HTMLの基本構造確認
        assert "<!DOCTYPE html>" in report.content
        assert "テストレポート" in report.content
        assert "USDJPY" in report.content


def test_generate_json_report(report_generator, sample_backtest_result, sample_config, temp_dir):
    """JSONレポート生成テスト"""
    sample_config.format = ReportFormat.JSON
    sample_config.output_path = f"{temp_dir}/test_report.json"
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analysis = Mock()
        mock_analysis.performance_score = Mock()
        mock_analysis.performance_score.overall_score = 78.5
        mock_analysis.performance_score.rating = PerformanceRating.GOOD
        mock_analysis.performance_score.strengths = ["高い利益率"]
        mock_analysis.performance_score.weaknesses = ["高いリスク"]
        mock_analysis.performance_score.recommendations = ["リスク軽減"]
        mock_analysis.risk_metrics = Mock()
        mock_analysis.risk_metrics.max_drawdown = 0.03
        mock_analysis.risk_metrics.volatility = 0.15
        mock_analysis.risk_metrics.value_at_risk_95 = -0.02
        mock_analysis.risk_metrics.downside_deviation = 0.08
        mock_analysis.return_metrics = Mock()
        mock_analysis.return_metrics.total_return = 0.05
        mock_analysis.return_metrics.annualized_return = 0.06
        mock_analysis.return_metrics.monthly_win_rate = 0.75
        mock_analysis.ratio_metrics = Mock()
        mock_analysis.ratio_metrics.sharpe_ratio = 1.8
        mock_analysis.ratio_metrics.sortino_ratio = 2.1
        mock_analysis.trading_metrics = Mock()
        mock_analysis.trading_metrics.profit_factor = 1.67
        mock_analysis.trading_metrics.recovery_factor = 1.5
        mock_analysis.trading_metrics.trade_efficiency = 0.7
        mock_analysis.trading_metrics.average_holding_period = 12.0
        mock_analysis.backtest_result = sample_backtest_result
        mock_analyze.return_value = mock_analysis
        
        report = report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)
        
        assert isinstance(report, GeneratedReport)
        assert report.config.format == ReportFormat.JSON
        
        # JSONとして有効か確認
        json_data = json.loads(report.content)
        assert 'report_info' in json_data
        assert 'backtest_config' in json_data
        assert 'performance_metrics' in json_data
        assert json_data['report_info']['title'] == "テストレポート"


def test_generate_csv_report(report_generator, sample_backtest_result, sample_config, temp_dir):
    """CSVレポート生成テスト"""
    sample_config.format = ReportFormat.CSV
    sample_config.output_path = f"{temp_dir}/test_report.csv"
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analysis = Mock()
        mock_analysis.backtest_result = sample_backtest_result
        mock_analyze.return_value = mock_analysis
        
        report = report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)
        
        assert isinstance(report, GeneratedReport)
        assert report.config.format == ReportFormat.CSV
        
        # CSVヘッダーの確認
        lines = report.content.split('\n')
        header_line = lines[0]
        assert 'Trade_ID' in header_line
        assert 'Symbol' in header_line
        assert 'Position_Type' in header_line
        assert 'Net_Profit' in header_line


def test_generate_csv_report_no_trades(report_generator, sample_backtest_config, sample_config, temp_dir):
    """CSVレポート生成テスト - 取引なし"""
    sample_config.format = ReportFormat.CSV
    sample_config.output_path = f"{temp_dir}/test_report.csv"
    
    # 取引のない結果
    empty_result = BacktestResult(
        config=sample_backtest_config,
        status=BacktestStatus.COMPLETED,
        metrics=BacktestMetrics(total_trades=0, winning_trades=0, losing_trades=0, win_rate=0.0),
        positions=[],
        equity_curve=pd.DataFrame(),
        execution_time=0.1
    )
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analysis = Mock()
        mock_analysis.backtest_result = empty_result
        mock_analyze.return_value = mock_analysis
        
        report = report_generator.generate_single_strategy_report(empty_result, sample_config)
        
        assert "No trades to export" in report.content


def test_generate_comparison_report(report_generator, sample_backtest_result, sample_config, temp_dir):
    """比較レポート生成テスト"""
    sample_config.report_type = ReportType.STRATEGY_COMPARISON
    sample_config.format = ReportFormat.HTML
    sample_config.output_path = f"{temp_dir}/comparison_report.html"
    
    # 複数の結果を作成
    results = [sample_backtest_result, sample_backtest_result]
    
    with patch.object(report_generator.performance_analyzer, 'compare_strategies') as mock_compare:
        mock_compare.return_value = {
            'summary': {'total_strategies': 2},
            'strategy_comparison': [
                {
                    'strategy_id': 'Strategy_1',
                    'overall_score': 78.5,
                    'total_return': 0.05,
                    'max_drawdown': 0.03,
                    'sharpe_ratio': 1.8,
                    'win_rate': 0.6
                },
                {
                    'strategy_id': 'Strategy_2',
                    'overall_score': 65.2,
                    'total_return': 0.03,
                    'max_drawdown': 0.05,
                    'sharpe_ratio': 1.2,
                    'win_rate': 0.55
                }
            ],
            'rankings': {},
            'best_overall': {'strategy_id': 'Strategy_1', 'overall_score': 78.5}
        }
        
        with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
            mock_analysis = Mock()
            mock_analyze.return_value = mock_analysis
            
            report = report_generator.generate_comparison_report(results, sample_config)
            
            assert isinstance(report, GeneratedReport)
            assert "比較" in report.content or "comparison" in report.content.lower()


def test_generate_optimization_report(report_generator, sample_backtest_result, sample_config, temp_dir):
    """最適化レポート生成テスト"""
    sample_config.report_type = ReportType.OPTIMIZATION_REPORT
    sample_config.format = ReportFormat.HTML
    sample_config.output_path = f"{temp_dir}/optimization_report.html"
    
    # 最適化結果
    optimization_results = [sample_backtest_result] * 3
    parameter_ranges = {
        'risk_per_trade': [0.01, 0.02, 0.03],
        'min_signal_confidence': [0.6, 0.7, 0.8]
    }
    
    with patch.object(report_generator, '_analyze_optimization_results') as mock_analyze_opt:
        mock_analyze_opt.return_value = {
            'summary': {
                'total_combinations': 9,
                'best_profit': 1000.0,
                'avg_profit': 500.0,
                'profit_std': 200.0,
                'avg_win_rate': 0.65,
                'avg_drawdown': 0.04
            },
            'best_configuration': {
                'config': {'risk_per_trade': 0.02},
                'metrics': {
                    'total_profit': 1000.0,
                    'win_rate': 0.7,
                    'profit_factor': 2.0,
                    'max_drawdown': 0.03,
                    'total_trades': 10
                }
            },
            'worst_configuration': {
                'config': {'risk_per_trade': 0.01},
                'metrics': {
                    'total_profit': 200.0,
                    'win_rate': 0.5,
                    'profit_factor': 1.2,
                    'max_drawdown': 0.05,
                    'total_trades': 8
                }
            },
            'parameter_sensitivity': {},
            'top_10_results': [
                {
                    'rank': 1,
                    'config': {'risk_per_trade': 0.02},
                    'total_profit': 1000.0,
                    'win_rate': 0.7,
                    'profit_factor': 2.0
                }
            ]
        }
        
        report = report_generator.generate_optimization_report(
            optimization_results, parameter_ranges, sample_config
        )
        
        assert isinstance(report, GeneratedReport)
        assert "最適化" in report.content or "optimization" in report.content.lower()


def test_save_report_default_path(report_generator, temp_dir):
    """レポート保存テスト - デフォルトパス"""
    content = "<html><body>Test Report</body></html>"
    config = ReportConfig(
        report_type=ReportType.SINGLE_STRATEGY,
        format=ReportFormat.HTML,
        title="Test"
    )
    
    # reportsディレクトリをtempディレクトリ内に作成
    reports_dir = Path(temp_dir) / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    with patch('pathlib.Path.cwd', return_value=Path(temp_dir)):
        file_path = report_generator._save_report(content, config)
        
        assert file_path.startswith("reports/")
        assert file_path.endswith(".html")
        
        # ファイルが存在することを確認
        full_path = Path(temp_dir) / file_path
        assert full_path.exists()
        
        # 内容確認
        with open(full_path, 'r', encoding='utf-8') as f:
            saved_content = f.read()
        assert saved_content == content


def test_save_report_custom_path(report_generator, temp_dir):
    """レポート保存テスト - カスタムパス"""
    content = '{"test": "data"}'
    custom_path = f"{temp_dir}/custom_report.json"
    config = ReportConfig(
        report_type=ReportType.SINGLE_STRATEGY,
        format=ReportFormat.JSON,
        title="Test",
        output_path=custom_path
    )
    
    file_path = report_generator._save_report(content, config)
    
    assert file_path == custom_path
    assert os.path.exists(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_content = f.read()
    assert saved_content == content


def test_analyze_optimization_results(report_generator, sample_backtest_result):
    """最適化結果分析テスト"""
    # 複数の結果を作成（異なる利益で）
    results = []
    profits = [1000, 800, 1200, 600, 900]
    
    for i, profit in enumerate(profits):
        result = BacktestResult(
            config=BacktestConfig(
                symbol="USDJPY",
                timeframe="H1",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 12, 31),
                initial_balance=100000.0,
                risk_per_trade=0.01 + i * 0.005  # 異なるリスク設定
            ),
            status=BacktestStatus.COMPLETED,
            metrics=BacktestMetrics(
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
                win_rate=0.6,
                total_profit=profit,
                profit_factor=1.5,
                max_drawdown=0.05
            ),
            positions=[],
            equity_curve=pd.DataFrame(),
            execution_time=1.0
        )
        results.append(result)
    
    parameter_ranges = {
        'risk_per_trade': [0.01, 0.015, 0.02, 0.025, 0.03]
    }
    
    with patch('numpy.mean') as mock_mean, patch('numpy.std') as mock_std:
        mock_mean.return_value = 900.0
        mock_std.return_value = 200.0
        
        analysis = report_generator._analyze_optimization_results(results, parameter_ranges)
        
        assert isinstance(analysis, dict)
        assert 'summary' in analysis
        assert 'best_configuration' in analysis
        assert 'worst_configuration' in analysis
        assert 'parameter_sensitivity' in analysis
        assert 'top_10_results' in analysis
        
        # 最高・最低利益の確認
        assert analysis['summary']['best_profit'] == 1200
        assert analysis['summary']['worst_profit'] == 600
        assert analysis['summary']['total_combinations'] == 5


def test_analyze_optimization_results_empty(report_generator):
    """最適化結果分析テスト - 空の結果"""
    analysis = report_generator._analyze_optimization_results([], {})
    
    assert 'error' in analysis
    assert analysis['error'] == 'No optimization results to analyze'


def test_template_generation(report_generator):
    """テンプレート生成テスト"""
    # デフォルトHTMLテンプレート
    html_template = report_generator._get_default_html_template()
    assert isinstance(html_template, str)
    assert "<!DOCTYPE html>" in html_template
    assert "{{ title }}" in html_template
    assert "{{ performance_score.overall_score }}" in html_template
    
    # 比較HTMLテンプレート
    comparison_template = report_generator._get_comparison_html_template()
    assert isinstance(comparison_template, str)
    assert "{{ title }}" in comparison_template
    assert "strategy_comparison" in comparison_template
    
    # 最適化HTMLテンプレート
    optimization_template = report_generator._get_optimization_html_template()
    assert isinstance(optimization_template, str)
    assert "{{ title }}" in optimization_template
    assert "optimization_analysis" in optimization_template


def test_unsupported_format_error(report_generator, sample_backtest_result, sample_config):
    """サポートされていない形式エラーテスト"""
    sample_config.format = "UNSUPPORTED"  # 無効な形式
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analyze.return_value = Mock()
        
        with pytest.raises(ValueError, match="サポートされていない形式"):
            report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)


def test_json_report_with_detailed_trades(report_generator, sample_backtest_result, sample_config, temp_dir):
    """JSON詳細取引レポートテスト"""
    sample_config.format = ReportFormat.JSON
    sample_config.include_detailed_trades = True
    sample_config.output_path = f"{temp_dir}/detailed_report.json"
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analysis = Mock()
        mock_analysis.backtest_result = sample_backtest_result
        mock_analysis.performance_score = Mock()
        mock_analysis.performance_score.overall_score = 78.5
        mock_analysis.performance_score.rating = PerformanceRating.GOOD
        mock_analysis.performance_score.strengths = []
        mock_analysis.performance_score.weaknesses = []
        mock_analysis.performance_score.recommendations = []
        mock_analysis.risk_metrics = Mock()
        mock_analysis.risk_metrics.max_drawdown = 0.03
        mock_analysis.risk_metrics.volatility = 0.15
        mock_analysis.risk_metrics.value_at_risk_95 = -0.02
        mock_analysis.risk_metrics.downside_deviation = 0.08
        mock_analysis.return_metrics = Mock()
        mock_analysis.return_metrics.total_return = 0.05
        mock_analysis.return_metrics.annualized_return = 0.06
        mock_analysis.return_metrics.monthly_win_rate = 0.75
        mock_analysis.ratio_metrics = Mock()
        mock_analysis.ratio_metrics.sharpe_ratio = 1.8
        mock_analysis.ratio_metrics.sortino_ratio = 2.1
        mock_analysis.trading_metrics = Mock()
        mock_analysis.trading_metrics.profit_factor = 1.67
        mock_analysis.trading_metrics.recovery_factor = 1.5
        mock_analysis.trading_metrics.trade_efficiency = 0.7
        mock_analysis.trading_metrics.average_holding_period = 12.0
        mock_analyze.return_value = mock_analysis
        
        report = report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)
        
        json_data = json.loads(report.content)
        assert 'trades' in json_data
        assert len(json_data['trades']) == len(sample_backtest_result.positions)
        
        # 取引データの構造確認
        first_trade = json_data['trades'][0]
        assert 'id' in first_trade
        assert 'symbol' in first_trade
        assert 'position_type' in first_trade
        assert 'net_profit' in first_trade


def test_monthly_breakdown_inclusion(report_generator, sample_backtest_result, sample_config, temp_dir):
    """月次詳細含有テスト"""
    sample_config.format = ReportFormat.JSON
    sample_config.include_monthly_breakdown = True
    sample_config.output_path = f"{temp_dir}/monthly_report.json"
    
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analysis = Mock()
        mock_analysis.backtest_result = sample_backtest_result
        mock_analysis.performance_score = Mock()
        mock_analysis.performance_score.overall_score = 78.5
        mock_analysis.performance_score.rating = PerformanceRating.GOOD
        mock_analysis.performance_score.strengths = []
        mock_analysis.performance_score.weaknesses = []
        mock_analysis.performance_score.recommendations = []
        mock_analysis.risk_metrics = Mock()
        mock_analysis.risk_metrics.max_drawdown = 0.03
        mock_analysis.risk_metrics.volatility = 0.15
        mock_analysis.risk_metrics.value_at_risk_95 = -0.02
        mock_analysis.risk_metrics.downside_deviation = 0.08
        mock_analysis.return_metrics = Mock()
        mock_analysis.return_metrics.total_return = 0.05
        mock_analysis.return_metrics.annualized_return = 0.06
        mock_analysis.return_metrics.monthly_win_rate = 0.75
        mock_analysis.ratio_metrics = Mock()
        mock_analysis.ratio_metrics.sharpe_ratio = 1.8
        mock_analysis.ratio_metrics.sortino_ratio = 2.1
        mock_analysis.trading_metrics = Mock()
        mock_analysis.trading_metrics.profit_factor = 1.67
        mock_analysis.trading_metrics.recovery_factor = 1.5
        mock_analysis.trading_metrics.trade_efficiency = 0.7
        mock_analysis.trading_metrics.average_holding_period = 12.0
        mock_analyze.return_value = mock_analysis
        
        report = report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)
        
        json_data = json.loads(report.content)
        assert 'monthly_returns' in json_data
        assert len(json_data['monthly_returns']) > 0


def test_file_encoding_utf8(report_generator, sample_config, temp_dir):
    """UTF-8エンコーディングテスト"""
    japanese_content = "<html><body>日本語テストレポート 株式会社</body></html>"
    config = ReportConfig(
        report_type=ReportType.SINGLE_STRATEGY,
        format=ReportFormat.HTML,
        title="日本語タイトル",
        output_path=f"{temp_dir}/japanese_report.html"
    )
    
    file_path = report_generator._save_report(japanese_content, config)
    
    # UTF-8で正しく読み書きできることを確認
    with open(file_path, 'r', encoding='utf-8') as f:
        saved_content = f.read()
    
    assert saved_content == japanese_content
    assert "日本語" in saved_content
    assert "株式会社" in saved_content


def test_error_handling_in_report_generation(report_generator, sample_backtest_result, sample_config):
    """レポート生成エラーハンドリングテスト"""
    with patch.object(report_generator.performance_analyzer, 'analyze_performance') as mock_analyze:
        mock_analyze.side_effect = Exception("分析エラー")
        
        with pytest.raises(Exception):
            report_generator.generate_single_strategy_report(sample_backtest_result, sample_config)