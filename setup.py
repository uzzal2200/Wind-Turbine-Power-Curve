from setuptools import setup, find_packages

setup(
    name="da-pinn",
    version="1.0.0",
    description="Degradation-Aware Physics-Informed Neural Network for Wind Turbine Power Curve Modeling",
    author="Md. Uzzal Mia, Sajib Debnath",
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "joblib>=1.3.0",
    ],
)
