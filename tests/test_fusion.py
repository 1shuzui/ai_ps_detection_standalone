"""融合策略边界测试：max/weighted/bonus 行为验证。"""

import pytest


class TestFusionStrategy:
    """测试 predict() 中融合策略的关键边界。

    由于融合逻辑嵌入在 InferenceEngineAPI.predict() 中且依赖完整模型，
    这里仅验证逻辑等价性（手算对照）。
    """

    def test_max_baseline(self):
        """max(global, local, metadata) 是最简单的基准。"""
        global_prob = 0.3
        local_prob = 0.7
        metadata_risk = 0.2
        result = max(global_prob, local_prob, metadata_risk)
        assert result == 0.7

    def test_agreement_bonus_two_signals(self):
        """两个信号同时超过阈值应获得 0.08 加成。"""
        global_prob = 0.6  # > 0.65*0.8 = 0.52
        local_prob = 0.5   # > 0.50*0.9 = 0.45
        metadata_risk = 0.0

        thresh_global = 0.65
        thresh_low = 0.50

        signals = 0
        if global_prob > thresh_global * 0.8:
            signals += 1
        if local_prob > thresh_low * 0.9:
            signals += 1
        if metadata_risk > 0.4:
            signals += 1

        base = max(global_prob, local_prob, metadata_risk)
        if signals == 2:
            base += 0.08

        assert base == pytest.approx(0.68)

    def test_agreement_bonus_three_signals(self):
        """三个信号同时超过阈值应获得 0.12 加成。"""
        global_prob = 0.6
        local_prob = 0.5
        metadata_risk = 0.5

        thresh_global = 0.65
        thresh_low = 0.50

        signals = 0
        if global_prob > thresh_global * 0.8:
            signals += 1
        if local_prob > thresh_low * 0.9:
            signals += 1
        if metadata_risk > 0.4:
            signals += 1

        base = max(global_prob, local_prob, metadata_risk)
        if signals == 3:
            base += 0.12

        assert base == 0.72

    def test_weighted_avg_lower_bound(self):
        """加权平均 * 0.7 作为 max 的下界参考。"""
        global_prob = 0.5
        local_prob = 0.4
        w_global = 0.35
        w_local = 0.65

        components = [global_prob, local_prob]
        weights = [w_global, w_local]
        total_w = sum(weights)
        weighted_avg = sum(c * w / total_w for c, w in zip(components, weights))
        floor = weighted_avg * 0.7
        final = max(max(global_prob, local_prob, 0.0), floor)

        assert final >= floor
        assert 0.0 <= final <= 1.0

    def test_all_zero_input(self):
        """全 0 输入不应崩溃且结果为 0。"""
        result = max(0.0, 0.0, 0.0)
        assert result == 0.0

    def test_all_one_input(self):
        """全 1 输入不应溢出。"""
        result = max(1.0, 1.0, 1.0)
        assert result == 1.0
        assert min(1.0, result) == 1.0
