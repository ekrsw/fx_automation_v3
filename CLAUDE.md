# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

FXの自動売買システム。ダウ理論とエリオット波動理論を基盤としたスイングトレード戦略を実装。MT5プラットフォームとPython FastAPIによるアーキテクチャで構築。**Phase 5（バックテストシステム）完了 - 80+テスト成功**。

## アーキテクチャ

### レイヤー構成
```
Presentation Layer (HTML/CSS/JS) ← REST API
↓
API Layer (FastAPI) → app/api/
↓
Business Logic Layer → app/services/
├── MT5Service (mt5_service.py) ✓
├── DataService (data_service.py) ✓
├── AnalysisEngine (analysis_engine.py) ✓
├── DowTheoryService (dow_theory.py) ✓
├── ElliottWaveService (elliott_wave.py) ✓
├── SignalGeneratorService (signal_generator.py) ✓
├── TechnicalIndicators (technical_indicators.py) ✓
├── TradingEngine (trading_engine.py) ✓
├── RiskManager (risk_manager.py) ✓
├── HistoricalDataService (historical_data_service.py) ✓
├── BacktestEngine (backtest_engine.py) ✓
├── PerformanceAnalyzer (performance_analyzer.py) ✓
└── ReportGenerator (report_generator.py) ✓
↓
Data Access Layer → app/models/ + SQLAlchemy
↓
MT5 Integration Layer → MetaTrader5 Python API
```

### 重要なコンポーネント

**app/services/historical_data_service.py**: 履歴データ取得・品質管理。MT5からの自動データ取得、重複・欠落検出、データ品質スコア算出。10通貨ペア×7タイムフレーム対応。

**app/services/backtest_engine.py**: 包括的バックテストエンジン。複数実行モード（SIMULATION/PAPER/LIVE）、ポジション管理、パラメータ最適化、エクイティカーブ生成。

**app/services/performance_analyzer.py**: 詳細パフォーマンス分析。100以上のメトリクス計算、4カテゴリ分析（リスク・リターン・比率・取引）、0-100点スコア算出。

**app/services/report_generator.py**: 多形式レポート生成。HTML/JSON/CSV対応、Jinja2テンプレート、matplotlib チャート生成、UTF-8完全サポート。

**app/services/trading_engine.py**: 取引エンジン。3実行モード対応、自動取引戦略、ポジション監視、リスク管理統合。

**app/services/risk_manager.py**: リスク管理。固定リスク戦略（資金2%）、ポートフォリオリスク評価、通貨集中リスク検証。

**app/services/analysis_engine.py**: 統合分析エンジン。全分析モジュールを統合し、設定可能なパラメータと柔軟な戦略変更を提供。

**app/services/dow_theory.py**: ダウ理論アルゴリズム。スイングポイント検出、高値・安値分類、トレンド方向判定を実装。Strategy Patternで複数戦略に対応。

**app/services/elliott_wave.py**: エリオット波動検出。5波推進パターンと3波修正パターンの自動認識、フィボナッチレベル計算、無効化レベル設定。

**app/services/signal_generator.py**: シグナル生成エンジン。ダウ理論+エリオット波動+RSIを統合した複合戦略。リスク・リワード比計算とストップロス/テイクプロフィット設定。

**app/services/technical_indicators.py**: テクニカル指標計算。ATR、ZigZag、移動平均、RSI。Factory Patternによる柔軟な指標生成。

**app/services/mt5_service.py**: MT5プラットフォームとの統合。価格データ取得、取引実行、ポジション管理、エラーハンドリング。

**app/services/data_service.py**: データベース操作とMT5データの永続化。重複防止、クエリ最適化。

**app/models/**: SQLAlchemyモデル定義
- `price_data.py`: OHLCV価格データ（分足データ）
- `trades.py`: 取引履歴
- `technical_analysis.py`: テクニカル分析結果
- `system_status.py`: システム状態管理
- `positions.py`: ポジション管理（PositionType, PositionStatus, PositionHistory）

**app/api/api_v1/endpoints/**:
- `backtest.py`: バックテスト用11エンドポイント（履歴データ取得、バックテスト実行、パフォーマンス分析、レポート生成）
- `trading.py`: 取引制御用11エンドポイント（ポジション管理、リスク管理、緊急制御）
- `data.py`: データ取得エンドポイント
- `health.py`: ヘルスチェック

**app/core/database.py**: SQLAlchemy設定と依存性注入のget_db()関数

**app/core/config.py**: Pydantic Settingsによる環境変数管理

## 開発コマンド

### 環境セットアップ
```bash
# 依存関係インストール
pip install -r requirements.txt

# 環境変数設定（.envファイル作成）
cp .env.example .env
# MT5_LOGIN, MT5_PASSWORD, MT5_SERVERを設定
```

### データベース管理
```bash
# マイグレーション作成
alembic revision --autogenerate -m "description"

# マイグレーション実行
alembic upgrade head

# マイグレーション履歴確認
alembic history
```

### アプリケーション実行
```bash
# 開発サーバー起動
python -m app.main

# または
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### テスト実行
```bash
# 全テスト実行（80+テスト成功）
python -m pytest tests/ -v

# Phase 5バックテスト関連テスト
python -m pytest tests/test_historical_data_service.py -v
python -m pytest tests/test_backtest_engine.py -v
python -m pytest tests/test_performance_analyzer.py -v
python -m pytest tests/test_report_generator.py -v
python -m pytest tests/test_backtest_api.py -v

# Phase 4取引エンジン関連テスト
python -m pytest tests/test_trading_engine.py -v
python -m pytest tests/test_risk_manager.py -v
python -m pytest tests/test_positions.py -v
python -m pytest tests/test_trading_api.py -v

# Phase 3分析エンジン関連テスト
python -m pytest tests/test_dow_theory.py -v
python -m pytest tests/test_elliott_wave.py -v
python -m pytest tests/test_signal_generator.py -v
python -m pytest tests/test_technical_indicators.py -v

# 特定テストファイル実行
python -m pytest tests/test_mt5_service.py -v

# 特定テスト関数実行
python -m pytest tests/test_models.py::test_price_data_model -v

# 静音モード（概要のみ）
python -m pytest tests/ -q

# カバレッジ付きテスト
python -m pytest tests/ --cov=app --cov-report=html
```

## 設計原則

1. **シンプルさ最優先**: 複雑な機能は段階的に追加。大規模な変更は避ける。
2. **段階的開発**: Phase 1(基盤構築) → Phase 2(データ層) → Phase 3(分析エンジン) → Phase 4(取引エンジン) → Phase 5(バックテスト) ✓ → Phase 6(UI/API実装)
3. **テスト駆動**: 全機能に対する包括的テスト。モック使用でMT5依存を排除。
4. **環境変数管理**: 設定の外部化。.envファイルによる環境固有設定。
5. **Strategy Pattern**: 分析アルゴリズムは戦略パターンで実装。後から容易に変更・拡張可能。
6. **Factory Pattern**: テクニカル指標は工場パターンで生成。柔軟な指標追加対応。

## 重要な制約事項

- **MT5依存**: 実稼働時はMetaTrader5がインストールされた環境が必要
- **SQLite使用**: 開発・テスト用。本番環境では別DBを検討
- **スイングトレード特化**: 長期保有前提（数日〜数週間）
- **UTF-8エンコーディング**: 日本語テキストファイルはUTF-8で保存

## 標準ワークフロー (Standard Workflow)

1. まず、docsディレクトリのドキュメントを全て読み込みます。ここにはこれから作成するアプリケーションの詳細設計が入っています。
2. 問題を徹底的に検討し、コードベースで関連ファイルを確認し、projectplan.md に計画を記述します。
3. 計画には、完了したらチェックマークを付けられる ToDo 項目のリストを含めます。
4. 作業を開始する前に、私に連絡して計画を確認します。
5. 次に、ToDo 項目の作業を開始し、完了したら完了マークを付けます。
6. 各ステップで、どのような変更を行ったか、概要を説明してください。
7. タスクとコードの変更は、できる限りシンプルにしてください。大規模で複雑な変更は避けたいと考えています。すべての変更は、コードへの影響を最小限に抑える必要があります。すべてはシンプルさが重要です。
8. 日本語テキストはUTF-8エンコーディングで保存するようにしてください。文字化けが起こる場合は、UTF-8エンコーディングでファイルを再作成してください。
9. 最後に、projectplan.md ファイルにレビューセクションを追加し、変更内容の概要とその他の関連情報を記載します。

## Phase 6への準備状況

### 完了した基盤
- **バックテストシステム**: 履歴データ取得、品質チェック、パフォーマンス分析、レポート生成完了
- **取引エンジン**: ポジション管理、リスク制御、3実行モード対応完了
- **分析エンジン**: 完全動作（ダウ理論、エリオット波動、シグナル生成）
- **データ層**: MT5連携、データベース永続化、履歴管理
- **API層**: バックテスト・取引制御・分析結果取得エンドポイント
- **テスト基盤**: 80+テスト成功、包括的カバレージ

### 次の実装対象（Phase 6: UI/API実装）
- フロントエンド開発（リアルタイム表示）
- 管理画面実装
- WebSocket通信（リアルタイムデータ）
- ダッシュボード機能

## バックテストシステムアーキテクチャ (Phase 5完了)

### 履歴データ管理
```python
# HistoricalDataService - MT5からの自動データ取得
service = HistoricalDataService(db)
summary = service.fetch_and_store_historical_data("USDJPY", "H1", years_back=5)
quality_report = service.validate_historical_data("USDJPY", "H1")
```

### バックテストエンジン
```python
# BacktestEngine - 複数実行モード対応
engine = BacktestEngine(db)
config = BacktestConfig(symbol="USDJPY", timeframe="H1", 
                       execution_mode=ExecutionMode.SIMULATION)
result = engine.run_backtest(config)
optimization_results = engine.run_optimization(config, parameter_ranges)
```

### パフォーマンス分析
```python
# PerformanceAnalyzer - 100以上のメトリクス
analyzer = PerformanceAnalyzer()
analysis = analyzer.analyze_performance(backtest_result)
# 4カテゴリ分析: リスク、リターン、比率、取引メトリクス
# 0-100点スコア: EXCELLENT/GOOD/FAIR/POOR評価
```

### レポート生成
```python
# ReportGenerator - 多形式対応
generator = ReportGenerator()
config = ReportConfig(format=ReportFormat.HTML, include_charts=True)
report = generator.generate_single_strategy_report(result, config)
# HTML/JSON/CSV形式、チャート自動生成、UTF-8対応
```

## 取引エンジンアーキテクチャ (Phase 4完了)

### ポジション管理
```python
# Position モデル - ライフサイクル管理
position = Position(symbol="USDJPY", position_type=PositionType.BUY)
# PENDING → OPEN → CLOSED ステート管理
# 含み損益リアルタイム計算、リスクリワード比算出
```

### リスク管理
```python
# RiskManager - 固定リスク戦略
risk_manager = RiskManagerService()
calculation = risk_manager.calculate_position_size(balance=100000, risk_amount=2000)
risk_assessment = risk_manager.assess_portfolio_risk(positions)
# 4段階リスクレベル: LOW/MEDIUM/HIGH/CRITICAL
```

### 取引実行
```python
# TradingEngine - 3実行モード
engine = TradingEngineService(db, execution_mode=ExecutionMode.SIMULATION)
summary = engine.execute_signal(signal)
# LIVE/SIMULATION/PAPER モード切り替え
# 自動ポジション監視、リスク管理統合
```

## 分析エンジンアーキテクチャ (Phase 3完了)

### 戦略パターンの実装
```python
# 各分析モジュールは戦略を簡単に変更可能
dow_service.set_strategy(ClassicDowTheory(zigzag_deviation=3.0))
elliott_service.set_strategy(ClassicElliottWave(fib_ratios=custom_ratios))
signal_service.set_strategy(DowElliottCombinedStrategy(min_confidence=0.8))
```

### 主要クラス設計
- **TechnicalIndicator**: 全指標の基底クラス（ATR、ZigZag、RSI、MA）
- **DowTheoryStrategy**: ダウ理論戦略の基底クラス
- **ElliottWaveStrategy**: エリオット波動戦略の基底クラス  
- **SignalStrategy**: シグナル生成戦略の基底クラス

### データフロー
```
PriceData → TechnicalIndicators → DowTheory → ElliottWave → SignalGenerator → TradingSignal → TradingEngine → Position
```

## 重要な実装詳細

### 統合分析パターン
```python
# AnalysisEngineService - 全分析モジュールの協調動作
analysis_engine = AnalysisEngineService(db)
result = analysis_engine.analyze_symbol("USDJPY", "H4")

# 戦略の動的変更
analysis_engine.dow_service.set_strategy(ClassicDowTheory(zigzag_deviation=2.0))
analysis_engine.update_config({"min_data_points": 100})
```

### エラーハンドリングパターン
- データ不足時: `{"error": True, "message": "データ不足: X件 (最小Y件必要)"}`
- 分析成功時: `{"symbol": "USDJPY", "detailed_analysis": {...}, "overall_score": {...}}`
- 各分析モジュールは独立してエラーハンドリング

### バックテストAPIエンドポイント
```python
# 11個の専用エンドポイント
POST /api/v1/backtest/historical-data/fetch     # 履歴データ取得
GET  /api/v1/backtest/historical-data/overview  # データ概要取得
GET  /api/v1/backtest/historical-data/quality/{symbol}/{timeframe}  # 品質検証
POST /api/v1/backtest/backtest/run              # バックテスト実行
POST /api/v1/backtest/backtest/optimize         # パラメータ最適化
POST /api/v1/backtest/analysis/performance      # パフォーマンス分析
POST /api/v1/backtest/reports/generate          # レポート生成
GET  /api/v1/backtest/reports/download/{filename}  # レポートダウンロード
GET  /api/v1/backtest/backtest/presets          # プリセット設定
GET  /api/v1/backtest/system/status             # システム状態
```

### 取引APIエンドポイント
```python
# 11個の取引制御エンドポイント
POST /api/v1/trading/start                      # 取引開始
POST /api/v1/trading/stop                       # 取引停止
POST /api/v1/trading/emergency-close            # 緊急全クローズ
POST /api/v1/trading/positions/open             # ポジション手動オープン
POST /api/v1/trading/positions/{id}/close       # ポジションクローズ
POST /api/v1/trading/signals/execute            # シグナル手動実行
GET  /api/v1/trading/status                     # 取引状態取得
GET  /api/v1/trading/positions                  # ポジション一覧
GET  /api/v1/trading/risk-summary               # リスクサマリー
PUT  /api/v1/trading/mode/{mode}                # 実行モード変更
GET  /api/v1/trading/monitor                    # 監視データ取得
```

### ZigZagアルゴリズム修正済み
- ピーク検出ロジックが修正済み（technical_indicators.py:49-90）
- `elif` → `if` 変更により双方向チェック対応
- Dow理論の条件緩和（min_data_points: 10→3, min_swing_points: 4→2）

### テストアーキテクチャ
- Mock使用でMT5依存を完全排除
- 確定的トレンドパターンでテストの安定性確保
- Strategy/Factory Patternの柔軟性をテストで検証
- 80+テストによる包括的カバレージ

### フィボナッチ比率と波動ルール
```python
# Elliott Wave - 厳格なルール検証
- 波2は波1開始点の100%を下回らない
- 波3は最短波動ではない
- 波4は波1の価格領域に重複しない

# フィボナッチレベル自動計算
fib_levels = strategy._calculate_fibonacci_levels(start, end, WaveLabel.WAVE_2)
```

### データ品質管理
```python
# DataQualityReport - 5段階品質評価
GOOD, HAS_GAPS, HAS_DUPLICATES, INVALID_FORMAT, MISSING_DATA
# 0-100点品質スコア算出
# 自動クリーニング・重複除去機能
```