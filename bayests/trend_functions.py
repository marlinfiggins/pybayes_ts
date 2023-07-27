import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from bayests.TimeSeriesTree import TimeSeriesNode


class FlatTrend(TimeSeriesNode):
    def __init__(self, name=None):
        self.name = name or "FlatTrend()"

    @staticmethod
    def _design(t):
        return jnp.ones_like(t)

    def design(self, t):
        self.features = self._design(t)

    def _model(self, t):
        beta = numpyro.sample(self.name + "_beta", dist.Normal(0.0, 1.0))
        self.design(t)
        self.output = self.features * beta


class ConstantTrend(TimeSeriesNode):
    def __init__(self, k, s=None, name=None, intercept=True):
        assert k is not None or s is not None, (
            "Constant trend must have either given a num of change points `k`"
            + "or change points `s` directly."
        )

        self.s = s  # Location of change points # If None, we can update later
        self.k = k if s is None else len(s)  # Number of change points
        self.name = name or f"ConstantTrend(k={k})"
        self.intercept = intercept

    @staticmethod
    def _design(s, t, intercept):
        design = t[:, None] > s
        if intercept:
            design = jnp.hstack((design, jnp.ones_like(t)[:, None]))
        return design

    def design(self, t):
        if self.s is None:
            self.s = jnp.linspace(jnp.min(t), jnp.max(t), num=self.k)
        self.features = self._design(self.s, t, self.intercept)

    def _model(self, t):
        with numpyro.plate(self.name + f"_plate_{self.k}", self.k):
            beta = numpyro.sample(self.name + "_beta", dist.Normal(0.0, 1.0))
        self.design(t)
        self.output = jnp.dot(self.features, beta)


class LinearTrend(TimeSeriesNode):
    def __init__(
        self,
        k=None,
        s=None,
        name=None,
        intercept=True,
    ):
        assert k is not None or s is not None, (
            "Linear trend must have either given a num of change points `k`"
            + "or change points `s` directly."
        )
        super().__init__()
        self.s = s  # Location of change points # If None, we can update later
        self.k = k if s is None else len(s)  # Number of change points
        self.name = name or f"LinearTrend(k={k})"
        self.intercept = intercept

    @staticmethod
    def _design(s, t, intercept):
        design = (t[:, None] - s) * (t[:, None] > s)
        if intercept:
            design = jnp.hstack((design, jnp.ones_like(t)[:, None]))
        return design

    def design(self, t):
        if self.s is None:
            self.s = jnp.linspace(jnp.min(t), jnp.max(t), num=self.k)
        self.features = self._design(self.s, t, self.intercept)
        return self.features

    def _model(self, t):
        self.n_params = self.k + int(self.intercept)
        with numpyro.plate(self.name + f"_plate_{self.k}", self.n_params):
            beta = numpyro.sample(self.name + "_beta", dist.Normal(0.0, 1.0))
        self.design(t)
        self.output = jnp.dot(self.features, beta)
