from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist


class ObservationLikelihood(ABC):
    @abstractmethod
    def model(self, data, pred_raw):
        pass


class Normal(ObservationLikelihood):
    def __init__(self, raw_sigma_sd=0.01):
        self.rssd = raw_sigma_sd

    def model(self, data, pred_raw):
        sigma = numpyro.sample("sigma", dist.HalfNormal(self.rssd))
        numpyro.sample("obs", dist.Normal(pred_raw, sigma), obs=data)


class NegativeBiniomial(ObservationLikelihood):
    def __init__(
        self, raw_alpha_sd=0.01
    ):  # Want option to do by multiple dimensions
        self.rasd = raw_alpha_sd

    def model(self, data, pred_raw):
        raw_alpha = numpyro.sample(
            "raw_alpha", dist.HalfNormal(0.1, self.rasd)
        )
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(
                jnp.exp(pred_raw), jnp.power(raw_alpha, -2)
            ),
            obs=data,
        )


class Poisson(ObservationLikelihood):
    def __init__(self):
        pass

    def model(self, data, pred_raw):
        numpyro.sample("obs", dist.Poisson(jnp.exp(pred_raw)), obs=data)


class Binomial(ObservationLikelihood):
    def __init__(self):
        pass

    def model(self, data, N, pred_raw):
        numpyro.sample("obs", dist.BinomialLogits(N, pred_raw), obs=data)
