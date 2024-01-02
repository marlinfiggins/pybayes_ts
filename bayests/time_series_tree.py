from abc import ABC, abstractmethod

from jax.random import PRNGKey
from numpyro.infer import Predictive

from bayests.likelihood_transform import Normal
from bayests.fitting_models import run_mcmc, run_svi, svi_predict
from bayests.utils import ComponentType

import numpyro


class TimeSeriesNode(ABC):
    def __init__(self):
        self._components = []
        self.features = None
        self.output = None
        self.component_type = ComponentType.OTHER

    @abstractmethod
    def _model(self, X):
        pass

    def __add__(self, other):
        return AdditiveTSNode(self, other)

    def __mul__(self, other):
        return MultiplicativeTSNode(self, other)

    def model(self, t, y=None, likelihood=None):

        # Loop over model components
        components = self._unravel()
        for comp in components:
            comp._model(t)
        pred = numpyro.deterministic("pred", self._eval())

        if likelihood is None:
            likelihood = Normal()

        likelihood.model(y, pred)  # Really this can differ by model.

    def fit_mle(self, t, y, max_iter=10_000, lr=1e-4, key=PRNGKey(1)):
        params, guide = run_svi(
            self.model, "AutoDelta", t, y, max_iter, lr, key
        )
        map_estimates = svi_predict(self.model, guide, params, 1, t)
        return params, map_estimates

    def fit_svi(
        self, t, y, num_samples, max_iter=10_000, lr=1e-4, key=PRNGKey(1)
    ):
        params, guide = run_svi(
            self.model, "AutoMultivariateNormal", t, y, max_iter, lr, key
        )
        svi_estimates = svi_predict(self.model, guide, params, num_samples, t)
        return params, svi_estimates

    def fit_mcmc(self, t, y, num_warmup, num_samples, key=PRNGKey(1)):
        samples = run_mcmc(self.model, t, y, num_warmup, num_samples, key)
        predictive = Predictive(self.model, posterior_samples=samples)
        pred_samples = predictive(key, t, y)
        return samples, pred_samples

    def _eval(self):
        """
        Return output variable given parameters and features.
        This will return None if model has not sampled parameters yet.
        """
        return self.output

    def _unravel(self):
        return [self]

    @property
    def components(self):
        if self._components == []:
            self._components = self._unravel()
        return self._components

    @property
    def n_components(self):
        return len(self.components)


class TrendNode(TimeSeriesNode):
    def __init__(self):
        self.component_type = ComponentType.TREND


class SeasonalityNode(TimeSeriesNode):
    def __init__(self):
        self.component_type = ComponentType.SEASONALITY


class RegressorNode(TimeSeriesNode):
    def __init__(self):
        self.component_type = ComponentType.REGRESSOR


class CompositeNode(TimeSeriesNode):
    def __init__(self):
        self.component_type = ComponentType.COMPOSITE
        self._components = []


class AdditiveTSNode(CompositeNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def _eval(self):
        return self.left._eval() + self.right._eval()

    def _unravel(self):
        return self.left._unravel() + self.right._unravel()

    def _model(self):
        pass

    @property
    def n_components(self):
        return self.left.n_components + self.right.n_components


class MultiplicativeTSNode(CompositeNode):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def _eval(self):
        return self.left._eval() * self.right._eval()

    def _unravel(self):
        return self.left._unravel() + self.right._unravel()

    def _model(self):
        pass

    @property
    def n_components(self):
        return self.left.n_components + self.right.n_components
