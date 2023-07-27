from abc import ABC, abstractmethod

from bayests.LikelihoodTransform import Normal
from bayests.fitting_models import run_svi, svi_predict

import numpyro

# What overhauls are necessary for easy specification of hierarhcail models and partial pooling and running on multiple time series


class TimeSeriesNode(ABC):
    def __init__(self):
        self._components = []
        self.features = None

    @abstractmethod
    def _model(self, X):
        pass

    def __add__(self, other):
        return AdditiveTSNode(self, other)

    def __mul__(self, other):
        return MultiplicativeTSNode(self, other)

    def model(self, t, y, likelihood=None):

        # Loop over model components
        components = self._unravel()
        for comp in components:
            comp._model(t)
        pred = numpyro.deterministic("pred", self._eval())

        if likelihood is None:
            likelihood = Normal()

        likelihood.model(y, pred)  # Really this can differ by model.

    def fit_mle(self, t, y, max_iter=10_000):
        params, guide = run_svi(self.model, "AutoDelta", max_iter, t, y)
        map_estimates = svi_predict(self.model, guide, params, 1, t)
        return params, map_estimates

    def _eval(self):
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

    """ def get_features(self): """
    """     return [comp.features for comp in self.components] """


class AdditiveTSNode(TimeSeriesNode):
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


class MultiplicativeTSNode(TimeSeriesNode):
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


# In some cases, this can reduce to a matrix mult
# But I want to allow for more complicated models
# Evaluating a time series model requires matching up parameters with model

# Each model component will be evaluated with parameters
# Each component can produce its own fixed input data?
# We then tranverse the tree and compute the thingy

# For multiple time series we can vmap and then do the pooling seperately
# We'll have a model factory that takes in this tree and the data
