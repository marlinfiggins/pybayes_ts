from jax.random import PRNGKey
import numpyro
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
from numpyro.infer import MCMC, NUTS


def run_svi(model, guide_family, X, Y, max_iter, lr, key):
    if guide_family == "AutoDelta":
        guide = autoguide.AutoDelta(model)
    elif guide_family == "AutoDiagonalNormal":
        guide = autoguide.AutoDiagonalNormal(model)
    else:
        guide = autoguide.AutoMultivariateNormal(model)

    optimizer = numpyro.optim.Adam(lr)
    svi = SVI(model, guide, optimizer, Trace_ELBO())
    svi_results = svi.run(key, max_iter, t=X, y=Y)
    params = svi_results.params
    return params, guide


def svi_predict(model, guide, params, num_samples, t):
    predictive = Predictive(
        model=model, guide=guide, params=params, num_samples=num_samples
    )
    predictions = predictive(PRNGKey(1), t=t, y=None)
    return predictions


def run_mcmc(model, X, Y, num_warmup, num_samples, key):
    mcmc = MCMC(NUTS(model), num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(key, X, Y)
    return mcmc.get_samples()
