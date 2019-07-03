import numpy as np
import scipy as sp
import scipy.stats


class BetaBinomial(object):
    """
    Conjugate prior for a Binomial distribution with unknown parameter p.

    Parameters
    ----------
    a: prior for Beta(a, b) prior on parameter p of the distribution
    b: prior for Beta(a, b) prior on parameter p of the distribution
    """
    def __init__(self, a, b):
        self.__dict__.update({"a": a, "b": b})

    def fit(self, x):
        update_dict = {
            "a": self.a + np.sum(x),
            "b": self.b + len(x) - np.sum(x),
        }
        self.__dict__.update(update_dict)

    def get_posterior_prediction(self):
        return sp.stats.bernoulli(p=self.a / (self.a + self.b))


class GammaPoison(object):
    """
    Conjugate prior for a Poisson distribution with unknown rate λ.

    Parameters
    ----------
    alpha: prior for Gamma(alpha, beta) prior on parameter λ.
    beta:  prior for Gamma(alpha, beta) prior on parameter λ.
    """
    def __init__(self, alpha, beta):
        self.__dict__.update({"alpha": alpha, "beta": beta})

    def fit(self, x):
        update_dict = {
            "alpha": self.alpha + np.sum(x),
            "beta": self.beta + len(x)
        }
        self.__dict__.update(update_dict)

    def get_posterior_prediction(self):
        return sp.stats.negative_binomial(p=self.alpha / self.beta, n=self.beta)


class NormalInverseGamma(object):
    """"
    Conjugate prior for a univariate Normal distribution with unknown mean mu,
    variance sigma2.

    Parameters
    ----------
    m:    prior for N(m, tau2) on the mean mu of the distribution
    tau2: prior for N(m, tau2) on the mean mu of the distribution
    a:    prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:    prior for Γ(a, b) on the inverse sigma2 of the distribution
    """
    def __init__(self, m, tau2, a, b):
        self.__dict__.update({"m": m, "tau2": tau2, "a": a, "b": b})

    def fit(self, x):
        update_dict = {
            "m": (self.m / self.tau2 + len(x) * np.mean(x)) / \
                 (len(x) + 1 / self.tau2),
            "tau2": 1 / (len(x) + 1 / self.tau2),
            "a": self.a + len(x) / 2,
            "b": self.b + np.sum((x - np.mean(x)) ** 2) / 2 + \
                 (np.mean(x) - self.m) ** 2 * len(x) / self.tau2 / \
                 (len(x) + 1 / self.tau2) / 2,
        }
        self.__dict__.update(update_dict)

    def get_marginal_mu(self):
        return sp.stats.t(df=2 * self.a, loc=self.m,
                          scale=(self.tau2 * self.b / self.a) ** 0.5)

    def get_marginal_sigma2(self):
        return sp.stats.invgamma(self.a, scale=self.b)

    def get_posterior_prediction(self):
        return sp.stats.t(df=2 * self.a, loc=self.m,
                          scale=(self.b / self.a * (1 + self.tau2)) ** 0.5)


class NIGLinearRegression(object):
    """
    The normal inverse-gamma prior for a linear regression model with unknown
    variance and unknown relationship. Specifically,
        1/σ² ~ Γ(a, b)
        β ~ N(0, σ²V)

    Parameters
    ----------
    mu: prior for N(mu, v) on the model β
    v:  prior for N(mu, v) on the model β
    a:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    b:  prior for Γ(a, b) on the inverse sigma2 of the distribution
    """
    def __init__(self, mu, v, a, b):
        self.__dict__.update({"mu": mu, "v": v, "a": a, "b": b})

    def fit(self, x_tr, y_tr):
        m, _ = x_tr.shape
        mu_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr) @ \
                 (np.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr)
        v_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (y_tr - x_tr @ self.mu).T @ \
                np.linalg.inv(np.eye(m) + x_tr @ self.v @ x_tr.T) @ \
                (y_tr - x_tr @ self.mu.T)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        scales = np.array([x.T @ self.v @ x for x in x_te]) + 1
        scales = (self.b / self.a * scales) ** 0.5
        return sp.stats.t(df=2 * self.a, loc=x_te @ self.mu, scale=scales)

    def get_conditional_beta(self, sigma2):
        return sp.stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)

    def get_marginal_sigma2(self):
        return sp.stats.invgamma(self.a, scale=self.b)


class ZellnerGLinearRegression(object):
    """
    Zellner's g-prior for linear regression is specified as follows:
        σ² α 1/σ² [Jeffreys prior]
        β ~ N(0, σ²g(XᵀX)⁻¹)

    Parameters
    ----------
    g: prior for the N(0, σ²g(XᵀX)⁻¹) on the model β
    """
    def __init__(self, g):
        self.g = g

    def fit(self, x_tr, y_tr):
        m, _ = x_tr.shape
        mu_ast = self.g / (self.g + 1) * np.linalg.inv(x_tr.T @ x_tr) @ \
                 x_tr.T @ y_tr
        v_ast = self.g / (self.g + 1) * np.linalg.inv(x_tr.T @ x_tr)
        a_ast = 0.5 * m
        b_ast = 0.5 * (y_tr.T @ y_tr - self.g / (self.g + 1) * \
                y_tr.T @ x_tr @ np.linalg.inv(x_tr.T @ x_tr) @ x_tr.T @ y_tr)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        scales = np.array([x.T @ self.v @ x for x in x_te]) + 1
        scales = (self.b / self.a * scales) ** 0.5
        return sp.stats.t(df=2*self.a, loc=x_te @ self.mu, scale=scales)

    def get_conditional_beta(self, sigma2):
        return sp.stats.multivariate_normal(mean=self.mu, cov=sigma2 * self.v)

    def get_marginal_sigma2(self):
        return sp.stats.invgamma(self.a, scale=self.b)
