import numpy as np
import scipy as sp
import scipy.stats
from matplotlib import pyplot as plt
from conjugate_bayes.distributions import UnivariateNormalInverseGamma


if __name__ == "__main__":

    x = np.random.normal(loc=1, scale=1, size=20)
    model = UnivariateNormalInverseGamma(m=0, tau2=0.5, a=0.5, b=0.5)
    model.fit(x)

    mu = model.get_marginal_mu()
    sigma2 = model.get_marginal_sigma2()
    pred = model.get_posterior_prediction()
    mle = sp.stats.norm(loc=x.mean(), scale=x.std(ddof=1))

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    axis = np.linspace(-2, 2, 500)
    plt.plot(axis, mu.pdf(axis), color="black")
    plt.axvline(mu.mean(), color="black")
    plt.title("Marginal distribution over $\mu$")
    plt.subplot(1, 3, 2)
    axis = np.linspace(0, 2, 500)
    plt.plot(axis, sigma2.pdf(axis), color="black")
    plt.axvline(sigma2.mean(), color="black")
    plt.title("Marginal distribution over $\sigma^2$")
    plt.subplot(1, 3, 3)
    axis = np.linspace(-2, 2, 500)
    plt.plot(axis, pred.pdf(axis), color="black", label="Posterior")
    plt.title("Posterior predictive distribution")
    plt.plot(axis, mle.pdf(axis), color="blue", label="MLE")
    plt.plot(axis, sp.stats.norm(loc=1, scale=1).pdf(axis), color="green", label="True")
    plt.legend()
    plt.tight_layout()
    plt.show()
