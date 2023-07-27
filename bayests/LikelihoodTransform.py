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
