"""
setup.py
========
Installs the project in editable mode so src/ is resolved correctly.

Install with:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name             = "global_oil_risk_flask",
    version          = "3.0.0",
    description      = "Global Oil Supply Risk Prediction — Flask Web App (Worldwide Coverage)",
    python_requires  = ">=3.8",
    packages         = find_packages(),
    install_requires = [
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.1.0",
        "openpyxl>=3.0.0",
        "flask>=2.3.0",
        "werkzeug>=2.3.0",
        "jinja2>=3.1.0",
        "click>=8.0.0",
        "itsdangerous>=2.1.0",
        "markupsafe>=2.1.0",
        "blinker>=1.6.0",
    ],
    extras_require = {
        "xgboost": ["xgboost>=1.6.0"],
        "jupyter": ["jupyter", "notebook", "ipykernel", "jupyterlab"],
        "full":    ["xgboost>=1.6.0", "jupyter", "notebook", "ipykernel", "jupyterlab"],
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Framework :: Flask",
    ],
)
