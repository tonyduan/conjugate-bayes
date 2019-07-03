### Conjugate Bayesian Models

Last update: June 2019.

---

Lightweight Python library implementing a few conjugate Bayesian models. For details on the derivations see [1].

```
pip3 install conjugate-bayes
```

We support the following:

#### To fit distribution models

- Beta-Bernoulli
- Gamma-Poisson
- Normal Inverse-Gamma

#### To fit regression models

- Linear regression with Normal Inverse-Gamma prior
- Linear regression with Zellner's *g*-prior

#### Usage

Below we show an example fitting a simple Bayesian linear regression with unknown variance.

```python
Todo
```

The above example results in the following prediction intervals.

For further details the `examples/` folder.

#### References

[1] P. D. Hoff, A First Course in Bayesian Statistical Methods (New York: Springer-Verlag, 2009).

#### License

This library is available under the MIT License.