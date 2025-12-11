# acedcor (ACE + Distance Correlation)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**acedcor** is a Python package designed to detect and diagnose nonlinear relationships between variables. It combines the non-parametric **Alternating Conditional Expectation (ACE)** algorithm with **Distance Correlation (dCor)** to identify hidden dependencies that traditional metrics like Pearson correlation often miss.

Key Metric: **Delta dCor** (ΔdCor = dCor_After - dCor_Before). A high **ΔdCor** serves as a diagnostic signal for symmetric, non-linear relationships.

## Features

- **Unified Interface**: Perform ACE transformation and dCor calculation in a single function.
- **Diagnostic Power**: Use **ΔdCor** to distinguish between linear/monotonic and complex symmetric relationships.
- **Visualization**: Easily integrate with matplotlib to visualize transformations.
- **Robustness**: Validated against heavy-tailed and skewed error distributions.

## Prerequisites

This package requires a working **R** installation (version >= 4.0) with the following packages installed:
- `acepack`
- `energy`

This requirement exists because `acedcor` leverages the gold-standard statistical implementations of these algorithms via `rpy2`.

```R
# Run inside your R console
install.packages(c("acepack", "energy"))
```

## Installation

You can install `acedcor` using pip (ensure R and rpy2 prerequisites are met):

```bash
pip install acedcor
```
*(Note: If installing from source, run `pip install .` in the root directory)*

## Usage

```python
import numpy as np
from acedcor import calculate_dcor_improvement

# 1. Generate Data (e.g., Parabola y = x^2)
np.random.seed(42)
x = np.random.uniform(-1, 1, 500)
y = x**2 + np.random.normal(0, 0.1, 500)

# 2. Calculate Diagnostics
results = calculate_dcor_improvement(x, y, verbose=True)

# 3. Inspect Results
print(f"dCor Before: {results['dcor_before']:.4f}") 
# Expected: Low (~0.1-0.2) because dCor on raw parabola is weak? 
# Actually dCor detects dependence (approx 0.4-0.5), but Pearson is ~0.
print(f"dCor After:  {results['dcor_after']:.4f}")  
# Expected: High (~0.9-1.0) linearity after transformation
print(f"Improvement: {results['delta_dcor']:.4f}")
```

## Citation

If you use this software in your research, please cite:

> S. M. Park and H.-M. Kim, "acedcor: A Python package for detecting and diagnosing nonlinear relationships using ACE and Distance Correlation," *SoftwareX* (Submitted), 2025.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
