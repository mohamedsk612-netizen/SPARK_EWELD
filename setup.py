from setuptools import setup, find_packages

setup(
    name="spark",
    version="1.0.0",
    description="SPARK Pipeline — Extreme Weather Events Load Dataset",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "scikit-learn>=1.3",
        "openpyxl>=3.1",
    ],
    extras_require={
        "forecast": ["torch>=2.0"],
        "explain":  ["catboost>=1.2", "shap>=0.43"],
        "all":      ["torch>=2.0",
                     "catboost>=1.2", "shap>=0.43"],
    },
    entry_points={
        "console_scripts": [
            "spark=run:main",
        ],
    },
)

