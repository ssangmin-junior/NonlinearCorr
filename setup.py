from setuptools import setup, find_packages

setup(
    name="acedcor",
    version="1.0.0",
    description="A Python package for detecting and diagnosing nonlinear relationships using ACE and Distance Correlation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Sang Min Park",
    author_email="sm123012@naver.com",
    url="https://github.com/ssangmin-junior/NonlinearCorr",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.8',
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.0.0",
        "scipy>=1.5.0",
        "rpy2>=3.4.0",
        "matplotlib>=3.3.0",
    ],
    keywords="statistics nonlinear-correlation ace-algorithm distance-correlation",
)
