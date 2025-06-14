import pytest
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.services.risk_manager import (
    RiskManagerService, FixedRiskStrategy, RiskLevel,
    RiskAssessment, PositionSizeCalculation
)
from app.models.positions import Position, PositionType, PositionStatus, Base


@pytest.fixture
def db_session():
    """テスト用データベースセッション"""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture
def risk_manager(db_session):
    """リスク管理サービス"""
    return RiskManagerService(db_session)


@pytest.fixture
def fixed_risk_strategy():
    """固定リスク戦略"""
    return FixedRiskStrategy(risk_percentage=0.02, max_risk_percentage=0.05)


def test_fixed_risk_strategy_position_size_calculation(fixed_risk_strategy):
    """固定リスク戦略のポジションサイズ計算テスト"""
    result = fixed_risk_strategy.calculate_position_size(
        account_balance=100000,
        entry_price=150.0,
        stop_loss=149.0,
        symbol="USDJPY"
    )
    
    assert isinstance(result, PositionSizeCalculation)
    assert result.risk_amount == 2000.0  # 100000 * 0.02
    assert result.risk_percentage == 0.02
    assert result.recommended_size > 0
    assert result.max_size > result.recommended_size
    assert len(result.reasoning) > 0


def test_fixed_risk_strategy_no_stop_loss(fixed_risk_strategy):
    """ストップロス未設定時のテスト"""
    result = fixed_risk_strategy.calculate_position_size(
        account_balance=100000,
        entry_price=150.0,
        stop_loss=150.0,  # エントリー価格と同じ = ストップロス幅ゼロ
        symbol="USDJPY"
    )
    
    assert result.recommended_size == 0.0
    assert result.max_size == 0.0
    assert "ストップロスが設定されていません" in result.reasoning


def test_fixed_risk_strategy_pip_value_calculation(fixed_risk_strategy):
    """ピップ値計算テスト"""
    # JPYペア
    jpy_pip = fixed_risk_strategy._calculate_pip_value("USDJPY", 150.0)
    assert jpy_pip == 0.01
    
    # その他のペア
    eur_pip = fixed_risk_strategy._calculate_pip_value("EURUSD", 1.2000)
    assert eur_pip == 0.0001


def test_fixed_risk_strategy_assess_risk_empty_positions(fixed_risk_strategy):
    """ポジションなしでのリスク評価"""
    assessment = fixed_risk_strategy.assess_risk(
        positions=[],
        market_data={},
        account_balance=100000
    )
    
    assert isinstance(assessment, RiskAssessment)
    assert assessment.risk_level == RiskLevel.LOW
    assert assessment.risk_score < 30
    assert len(assessment.warnings) == 0


def test_fixed_risk_strategy_assess_risk_with_positions(fixed_risk_strategy):
    """ポジションありでのリスク評価"""
    # オープンポジション作成
    positions = [
        Position(
            symbol="USDJPY",
            position_type=PositionType.BUY,
            status=PositionStatus.OPEN,
            lot_size=0.1,
            risk_amount=1000.0
        ),
        Position(
            symbol="EURUSD",
            position_type=PositionType.SELL,
            status=PositionStatus.OPEN,
            lot_size=0.2,
            risk_amount=1500.0
        )
    ]
    
    assessment = fixed_risk_strategy.assess_risk(
        positions=positions,
        market_data={},
        account_balance=100000
    )
    
    assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
    assert assessment.risk_factors['total_risk'] == 0.025  # (1000+1500)/100000
    assert assessment.risk_factors['position_count'] == 0.4  # 2/5


def test_fixed_risk_strategy_high_risk_scenario(fixed_risk_strategy):
    """高リスクシナリオのテスト"""
    # 最大ポジション数に達した状態
    positions = [
        Position(
            symbol=f"PAIR{i}",
            position_type=PositionType.BUY,
            status=PositionStatus.OPEN,
            lot_size=1.0,
            risk_amount=2000.0
        )
        for i in range(5)  # 最大ポジション数
    ]
    
    assessment = fixed_risk_strategy.assess_risk(
        positions=positions,
        market_data={},
        account_balance=100000
    )
    
    assert assessment.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    # 警告メッセージの存在確認（完全一致ではなく部分一致）
    warnings_text = ' '.join(assessment.warnings)
    assert "最大ポジション数" in warnings_text
    assert "総リスク" in warnings_text


def test_risk_manager_calculate_position_size(risk_manager):
    """リスク管理サービスのポジションサイズ計算"""
    result = risk_manager.calculate_position_size(
        symbol="USDJPY",
        entry_price=150.0,
        stop_loss=149.0,
        account_balance=50000
    )
    
    assert isinstance(result, PositionSizeCalculation)
    assert result.risk_amount == 1000.0  # 50000 * 0.02
    assert result.recommended_size > 0


def test_risk_manager_assess_portfolio_risk(risk_manager):
    """ポートフォリオリスク評価テスト"""
    with patch.object(risk_manager.db, 'query') as mock_query:
        # モックポジション設定
        mock_positions = [
            Mock(is_open=True, risk_amount=1000.0, symbol="USDJPY", lot_size=0.1),
            Mock(is_open=True, risk_amount=1500.0, symbol="EURUSD", lot_size=0.2)
        ]
        
        mock_query.return_value.filter.return_value.all.return_value = mock_positions
        
        assessment = risk_manager.assess_portfolio_risk(account_balance=100000)
        
        assert isinstance(assessment, RiskAssessment)
        assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]


def test_risk_manager_validate_new_position_valid(risk_manager):
    """新規ポジション妥当性検証 - 有効ケース"""
    result = risk_manager.validate_new_position(
        symbol="USDJPY",
        position_type="buy",
        lot_size=0.1,
        entry_price=150.0,
        stop_loss=149.0,
        account_balance=100000
    )
    
    assert result['is_valid'] is True
    assert 'position_size_calc' in result
    assert 'risk_assessment' in result


def test_risk_manager_validate_new_position_invalid_size(risk_manager):
    """新規ポジション妥当性検証 - 無効サイズ"""
    # まず推奨サイズを取得
    calc_result = risk_manager.calculate_position_size(
        symbol="USDJPY",
        entry_price=150.0,
        stop_loss=149.0,
        account_balance=10000
    )
    
    # 推奨サイズの10倍以上のサイズでテスト
    large_size = max(calc_result.max_size * 2, calc_result.recommended_size * 10)
    
    result = risk_manager.validate_new_position(
        symbol="USDJPY",
        position_type="buy",
        lot_size=large_size,
        entry_price=150.0,
        stop_loss=149.0,
        account_balance=10000
    )
    
    # サイズが大きすぎる場合、エラーまたは警告が出る
    assert result['is_valid'] is False or len(result['warnings']) > 0 or len(result['errors']) > 0


def test_risk_manager_validate_new_position_invalid_prices(risk_manager):
    """新規ポジション妥当性検証 - 無効価格"""
    result = risk_manager.validate_new_position(
        symbol="USDJPY",
        position_type="buy",
        lot_size=0.1,
        entry_price=-150.0,  # 負の価格
        stop_loss=149.0,
        account_balance=100000
    )
    
    assert result['is_valid'] is False
    assert "価格は正の値である必要があります" in result['errors']


def test_risk_manager_validate_new_position_small_stop_loss(risk_manager):
    """新規ポジション妥当性検証 - 小さすぎるストップロス"""
    result = risk_manager.validate_new_position(
        symbol="USDJPY",
        position_type="buy",
        lot_size=0.1,
        entry_price=150.0,
        stop_loss=149.99,  # 0.01の差 = 0.007%
        account_balance=100000
    )
    
    assert "ストップロスが小さすぎる可能性があります" in result['warnings']


def test_risk_manager_get_risk_summary(risk_manager):
    """リスク管理サマリー取得テスト"""
    with patch.object(risk_manager.db, 'query') as mock_query:
        mock_query.return_value.filter.return_value.all.return_value = []
        
        summary = risk_manager.get_risk_summary(account_balance=100000)
        
        assert 'total_positions' in summary
        assert 'total_risk_amount' in summary
        assert 'risk_level' in summary
        assert 'risk_score' in summary
        assert 'warnings' in summary


def test_risk_manager_set_strategy(risk_manager):
    """戦略変更テスト"""
    new_strategy = FixedRiskStrategy(risk_percentage=0.01)
    risk_manager.set_strategy(new_strategy)
    
    assert risk_manager.strategy == new_strategy
    assert risk_manager.strategy.risk_percentage == 0.01


def test_concentration_risk_calculation(fixed_risk_strategy):
    """通貨集中リスク計算テスト"""
    # 同一通貨ペアに集中したポジション
    positions = [
        Position(
            symbol="USDJPY",
            position_type=PositionType.BUY,
            status=PositionStatus.OPEN,
            lot_size=0.5,
            risk_amount=1000.0
        ),
        Position(
            symbol="USDJPY",
            position_type=PositionType.SELL,
            status=PositionStatus.OPEN,
            lot_size=0.3,
            risk_amount=800.0
        )
    ]
    
    assessment = fixed_risk_strategy.assess_risk(
        positions=positions,
        market_data={},
        account_balance=100000
    )
    
    # USD/JPY集中により高い集中リスク
    assert assessment.risk_factors['concentration'] > 0.5
    # 警告メッセージの部分一致確認
    warnings_text = ' '.join(assessment.warnings)
    assert "通貨集中リスク" in warnings_text


def test_risk_score_calculation_edge_cases():
    """リスクスコア計算の境界値テスト"""
    strategy = FixedRiskStrategy()
    
    # 低リスク
    low_risk_positions = [
        Position(
            symbol="USDJPY",
            position_type=PositionType.BUY,
            status=PositionStatus.OPEN,
            lot_size=0.01,
            risk_amount=100.0
        )
    ]
    
    assessment = strategy.assess_risk(
        positions=low_risk_positions,
        market_data={},
        account_balance=100000
    )
    
    assert assessment.risk_level in [RiskLevel.LOW, RiskLevel.MEDIUM]
    assert assessment.risk_score < 50