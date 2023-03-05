from unittest.mock import patch

import pytest

from mabby.stats import Metric, MetricTransform


class TestMetric:
    BASE_METRICS = [Metric.REGRET, Metric.REWARDS, Metric.OPTIMALITY]
    NON_BASE_METRICS = [Metric.CUM_REGRET, Metric.CUM_REWARDS]

    @pytest.fixture(params=list(Metric))
    def metric(self, request):
        return request.param

    @pytest.fixture(params=BASE_METRICS)
    def base_metric(self, request):
        return request.param

    @pytest.fixture(params=NON_BASE_METRICS)
    def non_base_metric(self, request):
        return request.param

    def test_repr_is_title_case(self, metric):
        metric_repr = repr(metric)
        assert metric_repr.istitle()
        assert "_" not in metric_repr

    def test_is_base_returns_true_for_base_metrics(self, base_metric):
        assert base_metric.is_base()

    def test_is_base_returns_true_for_non_base_metrics(self, non_base_metric):
        assert not non_base_metric.is_base()

    def test_base_returns_self_for_base_metrics(self, base_metric):
        assert base_metric.base == base_metric

    def test_base_returns_base_for_non_base_metrics(self, non_base_metric):
        assert non_base_metric.base.is_base()

    @pytest.mark.parametrize("metrics", [list(Metric), NON_BASE_METRICS, BASE_METRICS])
    def test_filter_base_returns_only_base_metrics(self, metrics):
        for metric in Metric.filter_base(metrics):
            assert metric in self.BASE_METRICS
            assert metric in metrics

    def test_transform_returns_values_for_base_metrics(self, mocker, base_metric):
        values = mocker.Mock()
        assert Metric.transform(base_metric, values) == values

    def test_transform_returns_transformed_values_for_non_base_metrics(
        self, mocker, non_base_metric
    ):
        values, return_values = mocker.Mock(), mocker.Mock()
        with patch.object(MetricTransform, "func", return_value=return_values) as func:
            transformed_values = Metric.transform(non_base_metric, values)
            func.assert_called_once_with(values)
            assert transformed_values == return_values
