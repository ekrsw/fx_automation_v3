import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
from jinja2 import Template
import json
from pathlib import Path

from app.services.backtest_engine import BacktestResult, BacktestConfig
from app.services.performance_analyzer import ComprehensiveAnalysis, PerformanceAnalyzer

logger = logging.getLogger(__name__)


class ReportFormat(str, Enum):
    """レポート形式"""
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    PDF = "pdf"


class ReportType(str, Enum):
    """レポートタイプ"""
    SINGLE_STRATEGY = "single_strategy"
    STRATEGY_COMPARISON = "strategy_comparison"
    OPTIMIZATION_REPORT = "optimization_report"
    EXECUTIVE_SUMMARY = "executive_summary"


@dataclass
class ReportConfig:
    """レポート設定"""
    report_type: ReportType
    format: ReportFormat
    title: str
    include_charts: bool = True
    include_detailed_trades: bool = False
    include_monthly_breakdown: bool = True
    custom_template: Optional[str] = None
    output_path: Optional[str] = None


@dataclass
class GeneratedReport:
    """生成されたレポート"""
    config: ReportConfig
    content: str
    file_path: Optional[str] = None
    generation_time: datetime = None
    file_size: Optional[int] = None


class ReportGenerator:
    """レポート生成エンジン"""
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        
        # テンプレートディレクトリ
        self.template_dir = Path(__file__).parent.parent / "templates"
        self.template_dir.mkdir(exist_ok=True)
        
        # デフォルトテンプレート作成
        self._create_default_templates()
    
    def generate_single_strategy_report(
        self,
        backtest_result: BacktestResult,
        config: ReportConfig
    ) -> GeneratedReport:
        """単一戦略レポート生成"""
        
        logger.info(f"単一戦略レポート生成開始: {config.format.value}")
        
        try:
            # パフォーマンス分析
            analysis = self.performance_analyzer.analyze_performance(backtest_result)
            
            # レポート生成
            if config.format == ReportFormat.HTML:
                content = self._generate_html_report(analysis, config)
            elif config.format == ReportFormat.JSON:
                content = self._generate_json_report(analysis, config)
            elif config.format == ReportFormat.CSV:
                content = self._generate_csv_report(analysis, config)
            else:
                raise ValueError(f"サポートされていない形式: {config.format}")
            
            # ファイル保存
            file_path = self._save_report(content, config)
            
            report = GeneratedReport(
                config=config,
                content=content,
                file_path=file_path,
                generation_time=datetime.now(),
                file_size=len(content.encode('utf-8'))
            )
            
            logger.info(f"レポート生成完了: {file_path}")
            return report
            
        except Exception as e:
            logger.error(f"レポート生成エラー: {str(e)}")
            raise
    
    def generate_comparison_report(
        self,
        backtest_results: List[BacktestResult],
        config: ReportConfig
    ) -> GeneratedReport:
        """戦略比較レポート生成"""
        
        logger.info(f"比較レポート生成開始: {len(backtest_results)}戦略")
        
        try:
            # 戦略比較分析
            comparison_data = self.performance_analyzer.compare_strategies(backtest_results)
            
            # 各戦略の詳細分析
            detailed_analyses = []
            for result in backtest_results:
                analysis = self.performance_analyzer.analyze_performance(result)
                detailed_analyses.append(analysis)
            
            # レポート生成
            if config.format == ReportFormat.HTML:
                content = self._generate_comparison_html(comparison_data, detailed_analyses, config)
            elif config.format == ReportFormat.JSON:
                content = self._generate_comparison_json(comparison_data, detailed_analyses, config)
            else:
                raise ValueError(f"比較レポートでサポートされていない形式: {config.format}")
            
            # ファイル保存
            file_path = self._save_report(content, config)
            
            report = GeneratedReport(
                config=config,
                content=content,
                file_path=file_path,
                generation_time=datetime.now(),
                file_size=len(content.encode('utf-8'))
            )
            
            logger.info(f"比較レポート生成完了: {file_path}")
            return report
            
        except Exception as e:
            logger.error(f"比較レポート生成エラー: {str(e)}")
            raise
    
    def generate_optimization_report(
        self,
        optimization_results: List[BacktestResult],
        parameter_ranges: Dict[str, List[float]],
        config: ReportConfig
    ) -> GeneratedReport:
        """最適化レポート生成"""
        
        logger.info(f"最適化レポート生成開始: {len(optimization_results)}結果")
        
        try:
            # 最適化分析
            optimization_analysis = self._analyze_optimization_results(
                optimization_results, parameter_ranges
            )
            
            # レポート生成
            if config.format == ReportFormat.HTML:
                content = self._generate_optimization_html(optimization_analysis, config)
            elif config.format == ReportFormat.JSON:
                content = self._generate_optimization_json(optimization_analysis, config)
            else:
                raise ValueError(f"最適化レポートでサポートされていない形式: {config.format}")
            
            # ファイル保存
            file_path = self._save_report(content, config)
            
            report = GeneratedReport(
                config=config,
                content=content,
                file_path=file_path,
                generation_time=datetime.now(),
                file_size=len(content.encode('utf-8'))
            )
            
            logger.info(f"最適化レポート生成完了: {file_path}")
            return report
            
        except Exception as e:
            logger.error(f"最適化レポート生成エラー: {str(e)}")
            raise
    
    def _generate_html_report(self, analysis: ComprehensiveAnalysis, config: ReportConfig) -> str:
        """HTMLレポート生成"""
        
        template_path = self.template_dir / "single_strategy_report.html"
        
        if config.custom_template:
            template_content = config.custom_template
        else:
            template_content = self._get_default_html_template()
        
        template = Template(template_content)
        
        # テンプレート変数準備
        template_vars = {
            'title': config.title,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'backtest_config': analysis.backtest_result.config,
            'metrics': analysis.backtest_result.metrics,
            'performance_score': analysis.performance_score,
            'risk_metrics': analysis.risk_metrics,
            'return_metrics': analysis.return_metrics,
            'ratio_metrics': analysis.ratio_metrics,
            'trading_metrics': analysis.trading_metrics,
            'executive_summary': analysis.executive_summary,
            'detailed_analysis': analysis.detailed_analysis,
            'include_charts': config.include_charts,
            'equity_curve_chart': analysis.equity_curve_chart if config.include_charts else None,
            'monthly_returns_chart': analysis.monthly_returns_chart if config.include_charts else None,
            'drawdown_chart': analysis.drawdown_chart if config.include_charts else None,
            'trade_distribution_chart': analysis.trade_distribution_chart if config.include_charts else None
        }
        
        # 詳細取引データ
        if config.include_detailed_trades:
            template_vars['positions'] = analysis.backtest_result.positions
        
        # 月次詳細
        if config.include_monthly_breakdown and analysis.backtest_result.monthly_returns is not None:
            template_vars['monthly_breakdown'] = analysis.backtest_result.monthly_returns.to_dict('records')
        
        return template.render(**template_vars)
    
    def _generate_json_report(self, analysis: ComprehensiveAnalysis, config: ReportConfig) -> str:
        """JSONレポート生成"""
        
        report_data = {
            'report_info': {
                'title': config.title,
                'generation_time': datetime.now().isoformat(),
                'report_type': config.report_type.value,
                'format': config.format.value
            },
            'backtest_config': {
                'symbol': analysis.backtest_result.config.symbol,
                'timeframe': analysis.backtest_result.config.timeframe,
                'start_date': analysis.backtest_result.config.start_date.isoformat(),
                'end_date': analysis.backtest_result.config.end_date.isoformat(),
                'initial_balance': analysis.backtest_result.config.initial_balance,
                'risk_per_trade': analysis.backtest_result.config.risk_per_trade
            },
            'performance_metrics': {
                'overall_score': analysis.performance_score.overall_score,
                'rating': analysis.performance_score.rating.value,
                'total_trades': analysis.backtest_result.metrics.total_trades,
                'win_rate': analysis.backtest_result.metrics.win_rate,
                'total_profit': analysis.backtest_result.metrics.total_profit,
                'profit_factor': analysis.backtest_result.metrics.profit_factor,
                'max_drawdown': analysis.risk_metrics.max_drawdown,
                'sharpe_ratio': analysis.ratio_metrics.sharpe_ratio,
                'sortino_ratio': analysis.ratio_metrics.sortino_ratio
            },
            'detailed_metrics': {
                'risk_metrics': {
                    'max_drawdown': analysis.risk_metrics.max_drawdown,
                    'volatility': analysis.risk_metrics.volatility,
                    'var_95': analysis.risk_metrics.value_at_risk_95,
                    'downside_deviation': analysis.risk_metrics.downside_deviation
                },
                'return_metrics': {
                    'total_return': analysis.return_metrics.total_return,
                    'annualized_return': analysis.return_metrics.annualized_return,
                    'monthly_win_rate': analysis.return_metrics.monthly_win_rate
                },
                'trading_metrics': {
                    'profit_factor': analysis.trading_metrics.profit_factor,
                    'recovery_factor': analysis.trading_metrics.recovery_factor,
                    'trade_efficiency': analysis.trading_metrics.trade_efficiency,
                    'average_holding_period': analysis.trading_metrics.average_holding_period
                }
            },
            'analysis_results': {
                'strengths': analysis.performance_score.strengths,
                'weaknesses': analysis.performance_score.weaknesses,
                'recommendations': analysis.performance_score.recommendations
            }
        }
        
        # 詳細取引データ
        if config.include_detailed_trades:
            report_data['trades'] = [
                {
                    'id': pos.id,
                    'symbol': pos.symbol,
                    'position_type': pos.position_type,
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                    'exit_time': pos.exit_time.isoformat() if pos.exit_time else None,
                    'entry_price': pos.entry_price,
                    'exit_price': pos.exit_price,
                    'lot_size': pos.lot_size,
                    'net_profit': pos.net_profit,
                    'exit_reason': pos.exit_reason
                }
                for pos in analysis.backtest_result.positions
            ]
        
        # 月次詳細
        if config.include_monthly_breakdown and analysis.backtest_result.monthly_returns is not None:
            report_data['monthly_returns'] = analysis.backtest_result.monthly_returns.to_dict('records')
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_csv_report(self, analysis: ComprehensiveAnalysis, config: ReportConfig) -> str:
        """CSVレポート生成（取引詳細）"""
        
        if not analysis.backtest_result.positions:
            return "No trades to export"
        
        # 取引データをDataFrameに変換
        trades_data = []
        for pos in analysis.backtest_result.positions:
            trades_data.append({
                'Trade_ID': pos.id,
                'Symbol': pos.symbol,
                'Position_Type': pos.position_type,
                'Entry_Time': pos.entry_time,
                'Exit_Time': pos.exit_time,
                'Entry_Price': pos.entry_price,
                'Exit_Price': pos.exit_price,
                'Lot_Size': pos.lot_size,
                'Stop_Loss': pos.stop_loss,
                'Take_Profit': pos.take_profit,
                'Profit_Loss': pos.profit_loss,
                'Commission': pos.commission,
                'Net_Profit': pos.net_profit,
                'Exit_Reason': pos.exit_reason,
                'Duration_Hours': (pos.exit_time - pos.entry_time).total_seconds() / 3600 if pos.exit_time else None
            })
        
        df = pd.DataFrame(trades_data)
        return df.to_csv(index=False)
    
    def _generate_comparison_html(self, comparison_data: Dict[str, Any], 
                                analyses: List[ComprehensiveAnalysis], config: ReportConfig) -> str:
        """比較HTMLレポート生成"""
        
        template_content = self._get_comparison_html_template()
        template = Template(template_content)
        
        template_vars = {
            'title': config.title,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'comparison_data': comparison_data,
            'analyses': analyses,
            'strategy_count': len(analyses)
        }
        
        return template.render(**template_vars)
    
    def _generate_comparison_json(self, comparison_data: Dict[str, Any], 
                                analyses: List[ComprehensiveAnalysis], config: ReportConfig) -> str:
        """比較JSONレポート生成"""
        
        report_data = {
            'report_info': {
                'title': config.title,
                'generation_time': datetime.now().isoformat(),
                'report_type': 'strategy_comparison',
                'strategy_count': len(analyses)
            },
            'comparison_summary': comparison_data['summary'],
            'strategy_comparison': comparison_data['strategy_comparison'],
            'rankings': comparison_data['rankings'],
            'best_overall_strategy': comparison_data['best_overall'],
            'detailed_strategies': [
                {
                    'strategy_id': f"Strategy_{i+1}",
                    'performance_score': analysis.performance_score.overall_score,
                    'rating': analysis.performance_score.rating.value,
                    'total_return': analysis.return_metrics.total_return,
                    'max_drawdown': analysis.risk_metrics.max_drawdown,
                    'sharpe_ratio': analysis.ratio_metrics.sharpe_ratio,
                    'profit_factor': analysis.trading_metrics.profit_factor,
                    'strengths': analysis.performance_score.strengths,
                    'weaknesses': analysis.performance_score.weaknesses
                }
                for i, analysis in enumerate(analyses)
            ]
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _generate_optimization_html(self, optimization_analysis: Dict[str, Any], config: ReportConfig) -> str:
        """最適化HTMLレポート生成"""
        
        template_content = self._get_optimization_html_template()
        template = Template(template_content)
        
        template_vars = {
            'title': config.title,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_analysis': optimization_analysis
        }
        
        return template.render(**template_vars)
    
    def _generate_optimization_json(self, optimization_analysis: Dict[str, Any], config: ReportConfig) -> str:
        """最適化JSONレポート生成"""
        
        report_data = {
            'report_info': {
                'title': config.title,
                'generation_time': datetime.now().isoformat(),
                'report_type': 'optimization_report'
            },
            'optimization_results': optimization_analysis
        }
        
        return json.dumps(report_data, indent=2, ensure_ascii=False)
    
    def _analyze_optimization_results(self, results: List[BacktestResult], 
                                    parameter_ranges: Dict[str, List[float]]) -> Dict[str, Any]:
        """最適化結果分析"""
        
        if not results:
            return {'error': 'No optimization results to analyze'}
        
        # 結果を性能順にソート
        sorted_results = sorted(results, 
                              key=lambda x: x.metrics.total_profit, reverse=True)
        
        best_result = sorted_results[0]
        worst_result = sorted_results[-1]
        
        # 統計分析
        profits = [r.metrics.total_profit for r in results]
        win_rates = [r.metrics.win_rate for r in results]
        drawdowns = [r.metrics.max_drawdown for r in results]
        
        # パラメータ感応度分析
        parameter_sensitivity = {}
        for param_name in parameter_ranges.keys():
            # 簡易的な感応度分析
            param_profits = {}
            for result in results:
                param_value = getattr(result.config, param_name, None)
                if param_value is not None:
                    if param_value not in param_profits:
                        param_profits[param_value] = []
                    param_profits[param_value].append(result.metrics.total_profit)
            
            # 各パラメータ値の平均性能
            param_avg_performance = {
                value: np.mean(profits_list) 
                for value, profits_list in param_profits.items()
            }
            
            parameter_sensitivity[param_name] = param_avg_performance
        
        return {
            'summary': {
                'total_combinations': len(results),
                'best_profit': best_result.metrics.total_profit,
                'worst_profit': worst_result.metrics.total_profit,
                'avg_profit': np.mean(profits),
                'profit_std': np.std(profits),
                'avg_win_rate': np.mean(win_rates),
                'avg_drawdown': np.mean(drawdowns)
            },
            'best_configuration': {
                'config': best_result.config.__dict__,
                'metrics': {
                    'total_profit': best_result.metrics.total_profit,
                    'win_rate': best_result.metrics.win_rate,
                    'profit_factor': best_result.metrics.profit_factor,
                    'max_drawdown': best_result.metrics.max_drawdown,
                    'total_trades': best_result.metrics.total_trades
                }
            },
            'worst_configuration': {
                'config': worst_result.config.__dict__,
                'metrics': {
                    'total_profit': worst_result.metrics.total_profit,
                    'win_rate': worst_result.metrics.win_rate,
                    'profit_factor': worst_result.metrics.profit_factor,
                    'max_drawdown': worst_result.metrics.max_drawdown,
                    'total_trades': worst_result.metrics.total_trades
                }
            },
            'parameter_sensitivity': parameter_sensitivity,
            'top_10_results': [
                {
                    'rank': i + 1,
                    'config': result.config.__dict__,
                    'total_profit': result.metrics.total_profit,
                    'win_rate': result.metrics.win_rate,
                    'profit_factor': result.metrics.profit_factor
                }
                for i, result in enumerate(sorted_results[:10])
            ]
        }
    
    def _save_report(self, content: str, config: ReportConfig) -> str:
        """レポート保存"""
        
        if config.output_path:
            file_path = config.output_path
        else:
            # デフォルトパス生成
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{config.report_type.value}_{timestamp}.{config.format.value}"
            file_path = f"reports/{filename}"
        
        # ディレクトリ作成
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # ファイル保存
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return file_path
    
    def _create_default_templates(self):
        """デフォルトテンプレート作成"""
        
        # 単一戦略HTMLテンプレート
        single_strategy_template = self._get_default_html_template()
        with open(self.template_dir / "single_strategy_report.html", 'w', encoding='utf-8') as f:
            f.write(single_strategy_template)
        
        # 比較HTMLテンプレート
        comparison_template = self._get_comparison_html_template()
        with open(self.template_dir / "comparison_report.html", 'w', encoding='utf-8') as f:
            f.write(comparison_template)
        
        # 最適化HTMLテンプレート
        optimization_template = self._get_optimization_html_template()
        with open(self.template_dir / "optimization_report.html", 'w', encoding='utf-8') as f:
            f.write(optimization_template)
    
    def _get_default_html_template(self) -> str:
        """デフォルトHTMLテンプレート"""
        
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Helvetica Neue', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }
        .header p {
            margin: 10px 0 0 0;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .metric-card h3 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.1em;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .performance-score {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 8px;
            margin: 20px 0;
        }
        .performance-score h2 {
            margin: 0;
            font-size: 3em;
        }
        .chart-container {
            margin: 30px 0;
            text-align: center;
        }
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .section {
            margin: 30px 0;
            padding: 20px;
            border: 1px solid #e9ecef;
            border-radius: 8px;
        }
        .section h2 {
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        .strengths {
            background-color: #d4edda;
            border-color: #c3e6cb;
        }
        .weaknesses {
            background-color: #f8d7da;
            border-color: #f5c6cb;
        }
        .recommendations {
            background-color: #fff3cd;
            border-color: #ffeaa7;
        }
        ul {
            padding-left: 20px;
        }
        li {
            margin: 8px 0;
        }
        .footer {
            text-align: center;
            padding: 20px;
            background-color: #f8f9fa;
            color: #6c757d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>生成日時: {{ generation_time }}</p>
        </div>
        
        <div class="content">
            <!-- パフォーマンススコア -->
            <div class="performance-score">
                <h2>{{ performance_score.overall_score | round(1) }}/100</h2>
                <p>総合評価: {{ performance_score.rating.value | upper }}</p>
            </div>
            
            <!-- 主要メトリクス -->
            <div class="metric-grid">
                <div class="metric-card">
                    <h3>総取引数</h3>
                    <div class="metric-value">{{ metrics.total_trades }}</div>
                </div>
                <div class="metric-card">
                    <h3>勝率</h3>
                    <div class="metric-value">{{ (metrics.win_rate * 100) | round(1) }}%</div>
                </div>
                <div class="metric-card">
                    <h3>総利益</h3>
                    <div class="metric-value">{{ metrics.total_profit | round(2) }}</div>
                </div>
                <div class="metric-card">
                    <h3>プロフィットファクター</h3>
                    <div class="metric-value">{{ metrics.profit_factor | round(2) }}</div>
                </div>
                <div class="metric-card">
                    <h3>最大ドローダウン</h3>
                    <div class="metric-value">{{ (risk_metrics.max_drawdown * 100) | round(1) }}%</div>
                </div>
                <div class="metric-card">
                    <h3>シャープレシオ</h3>
                    <div class="metric-value">{{ ratio_metrics.sharpe_ratio | round(2) }}</div>
                </div>
            </div>
            
            <!-- チャート -->
            {% if include_charts %}
            {% if equity_curve_chart %}
            <div class="chart-container">
                <h3>エクイティカーブ</h3>
                <img src="data:image/png;base64,{{ equity_curve_chart }}" alt="Equity Curve">
            </div>
            {% endif %}
            {% endif %}
            
            <!-- 分析結果 -->
            {% if performance_score.strengths %}
            <div class="section strengths">
                <h2>強み</h2>
                <ul>
                {% for strength in performance_score.strengths %}
                    <li>{{ strength }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            {% if performance_score.weaknesses %}
            <div class="section weaknesses">
                <h2>改善点</h2>
                <ul>
                {% for weakness in performance_score.weaknesses %}
                    <li>{{ weakness }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            {% if performance_score.recommendations %}
            <div class="section recommendations">
                <h2>推奨事項</h2>
                <ul>
                {% for recommendation in performance_score.recommendations %}
                    <li>{{ recommendation }}</li>
                {% endfor %}
                </ul>
            </div>
            {% endif %}
        </div>
        
        <div class="footer">
            <p>FX自動売買システム - バックテストレポート</p>
            <p>このレポートは自動生成されました</p>
        </div>
    </div>
</body>
</html>
        """
    
    def _get_comparison_html_template(self) -> str:
        """比較HTMLテンプレート"""
        
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .comparison-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .comparison-table th, .comparison-table td { 
            border: 1px solid #ddd; padding: 8px; text-align: center; 
        }
        .comparison-table th { background-color: #f2f2f2; }
        .best { background-color: #d4edda; }
        .worst { background-color: #f8d7da; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>生成日時: {{ generation_time }}</p>
        <p>比較戦略数: {{ strategy_count }}</p>
    </div>
    
    <h2>戦略比較サマリー</h2>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>戦略</th>
                <th>総合スコア</th>
                <th>総リターン</th>
                <th>最大DD</th>
                <th>シャープレシオ</th>
                <th>勝率</th>
            </tr>
        </thead>
        <tbody>
        {% for strategy in comparison_data.strategy_comparison %}
            <tr>
                <td>{{ strategy.strategy_id }}</td>
                <td>{{ strategy.overall_score | round(1) }}</td>
                <td>{{ (strategy.total_return * 100) | round(1) }}%</td>
                <td>{{ (strategy.max_drawdown * 100) | round(1) }}%</td>
                <td>{{ strategy.sharpe_ratio | round(2) }}</td>
                <td>{{ (strategy.win_rate * 100) | round(1) }}%</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    
    <h2>最優秀戦略</h2>
    <p><strong>{{ comparison_data.best_overall.strategy_id }}</strong></p>
    <p>総合スコア: {{ comparison_data.best_overall.overall_score | round(1) }}/100</p>
</body>
</html>
        """
    
    def _get_optimization_html_template(self) -> str:
        """最適化HTMLテンプレート"""
        
        return """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .best-config { background: #d4edda; padding: 15px; border-radius: 8px; margin: 20px 0; }
        .optimization-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .optimization-table th, .optimization-table td { 
            border: 1px solid #ddd; padding: 8px; text-align: center; 
        }
        .optimization-table th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ title }}</h1>
        <p>生成日時: {{ generation_time }}</p>
    </div>
    
    <div class="summary">
        <h2>最適化サマリー</h2>
        <p>総組み合わせ数: {{ optimization_analysis.summary.total_combinations }}</p>
        <p>最高利益: {{ optimization_analysis.summary.best_profit | round(2) }}</p>
        <p>平均利益: {{ optimization_analysis.summary.avg_profit | round(2) }}</p>
    </div>
    
    <div class="best-config">
        <h2>最適設定</h2>
        <p>総利益: {{ optimization_analysis.best_configuration.metrics.total_profit | round(2) }}</p>
        <p>勝率: {{ (optimization_analysis.best_configuration.metrics.win_rate * 100) | round(1) }}%</p>
        <p>プロフィットファクター: {{ optimization_analysis.best_configuration.metrics.profit_factor | round(2) }}</p>
    </div>
    
    <h2>上位10結果</h2>
    <table class="optimization-table">
        <thead>
            <tr>
                <th>順位</th>
                <th>総利益</th>
                <th>勝率</th>
                <th>プロフィットファクター</th>
            </tr>
        </thead>
        <tbody>
        {% for result in optimization_analysis.top_10_results %}
            <tr>
                <td>{{ result.rank }}</td>
                <td>{{ result.total_profit | round(2) }}</td>
                <td>{{ (result.win_rate * 100) | round(1) }}%</td>
                <td>{{ result.profit_factor | round(2) }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
</body>
</html>
        """