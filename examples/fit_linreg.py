import numpy as np
import scipy as sp
import scipy.stats
from matplotlib import pyplot as plt
from conjugate_bayes.models import *


def gen_data(n=50, bound=1, deg=1, noise=0.1, intcpt=-0.5):
    x = np.linspace(-bound, bound, n)[:, np.newaxis]
    y = (x ** deg + noise * np.random.randn(*x.shape) + intcpt).squeeze()
    x = np.c_[np.ones_like(x), x]
    return x, y


if __name__ == "__main__":

    x_tr, y_tr = gen_data(500, deg=1, noise=0.3, bound=1)
    x_te, y_te = gen_data(500, deg=1, noise=0.3, bound=1)

    model = NIGLinearRegression(mu=np.zeros(2), v=100*np.eye(2), a=0.5, b=0.5)
    model.fit(x_tr, y_tr)

    sigma2 = model.get_marginal_sigma2()
    beta = model.get_conditional_beta(sigma2=sigma2.mean())

    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    axis1, axis2 = np.mgrid[-1:0:0.01, 0:2:0.01]
    plt.contourf(axis1, axis2, beta.logpdf(np.dstack((axis1, axis2))))
    plt.colorbar()
    plt.title("Conditional distribution over $\mu|E[\sigma^2]$")
    preds = model.predict(x_te)
    plt.subplot(1, 3, 2)
    axis = np.linspace(0, 0.5, 500)
    plt.plot(axis, sigma2.pdf(axis), color="black")
    plt.axvline(sigma2.mean(), color="black")
    plt.title("Marginal distribution over $\sigma^2$")
    preds = model.predict(x_te)
    plt.subplot(1, 3, 3)
    plt.plot(x_te, preds.mean(), color="black")
    plt.title("Prediction interval")
    plt.scatter(x_tr[:,1], y_tr, marker="x", color="grey")
    plt.plot(x_te, preds.mean() - 1.96 * preds.var() ** 0.5,
             linestyle="--", color="black")
    plt.plot(x_te, preds.mean() + 1.96 * preds.var() ** 0.5,
             linestyle="--", color="black")
    plt.xlim(0.01, 0.99)
    plt.tight_layout()
    plt.show()
