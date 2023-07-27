from jax.random import PRNGKey
import jax.numpy as jnp
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive


# TODO: Allow user to specify learning rate .etc
def run_svi(model, guide_family, max_iter, X, Y):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)
    else:
        guide = None

    optimizer = numpyro.optim.Adam(0.001)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(PRNGKey(1), max_iter, t=X, y=Y)
    params = svi_results.params
    return params, guide


def svi_predict(model, guide, params, num_samples, t):
    predictive = Predictive(
        model=model, guide=guide, params=params, num_samples=num_samples
    )
    predictions = predictive(PRNGKey(1), t=t, y=None)
    return predictions
