import numpy as np
import scipy as sp
import scipy.stats


class NIGLinearRegression(object):
    """
    The normal inverse-gamma prior.
    """
    def __init__(self, mu, v, a, b):
        self.__dict__.update({"mu": mu, "v": v, "a": a, "b": b})

    def fit(self, x_tr, y_tr):
        m, n = x_tr.shape
        mu_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr) @ \
                 (np.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr)
        v_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (self.mu.T @ np.linalg.inv(self.v) @ self.mu + \
                y_tr.T @ y_tr - mu_ast.T @ np.linalg.inv(v_ast) @ mu_ast)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})

    def predict(self, x_te):
        pass


class ZellnerGLinearRegression(object):
    """
    Zellner's g-prior specifies:
        β ~ N(0, σ²g(XᵀX)⁻¹)
        σ² ~ 1/σ²
    """
    def __init__(self, g):
        self.g = g

    def fit(self, x_tr, y_tr):
        mu_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr) @ \
                 (np.linalg.inv(self.v) @ self.mu + x_tr.T @ y_tr)
        v_ast = np.linalg.inv(np.linalg.inv(self.v) + x_tr.T @ x_tr)
        a_ast = self.a + 0.5 * m
        b_ast = self.b + 0.5 * (self.mu.T @ np.linalg.inv(self.v) @ self.mu + \
                y_tr.T @ y_tr - mu_ast.T @ np.linalg.inv(v_ast) @ mu_ast)
        self.__dict__.update({"mu": mu_ast, "v": v_ast, "a": a_ast, "b": b_ast})
