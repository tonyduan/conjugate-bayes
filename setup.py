import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="conjugate-bayes",
    version="0.0.1",
    author="Tony Duan",
    author_email="tonyduan@cs.stanford.edu",
    description="Conjugate Bayesian linear regression and distribution models in Python..",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonyduan/conjugate-bayes",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
