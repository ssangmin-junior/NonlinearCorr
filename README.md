# acedcor: Replication Code for SoftwareX Submission

This repository provides the complete Python code and data necessary to reproduce all figures and tables presented in the SoftwareX paper: **"acedcor: A Python package for detecting and diagnosing nonlinear relationships using ACE and Distance Correlation"**.

The primary file is `nonlinear.ipynb`, a Jupyter Notebook that uses the `acedcor` package to perform the analyses described in the paper.

## 📜 About the Paper

The `acedcor` package (the subject of the paper) is a Python library that integrates the Alternating Conditional Expectation (ACE) algorithm with distance correlation (dCor). This combination serves as a powerful tool to not only detect but also diagnose complex nonlinear dependencies between variables, which are often missed by standard linear methods like the Pearson correlation.

This notebook serves as the practical implementation and validation of the methodology presented.

## 🚀 How to Run

To run the analysis and generate the figures, you will need both Python and R environments correctly configured.

### 1. Prerequisites

You must have the following software installed:

* **Python (>=3.8)**
* **R (>=4.0)**
* **Python Packages:**
    ```bash
    pip install acedcor rpy2 numpy pandas matplotlib scipy scikit-learn jupyter
    ```
* **R Packages:**
    The `acedcor` library uses `rpy2` to interface with R. You must have the core R packages `acepack` and `energy` installed in your R environment. You can install them by running the following command in your R console:
    ```R
    install.packages(c("acepack", "energy"))
    ```

### 2. Execution

1.  **Clone the Repository:**
    ```bash
    git clone [Your-Repository-URL]
    cd [Your-Repository-Directory]
    ```

2.  **Download the Data:**
    Ensure the real-world dataset `Plant_1_Weather_Sensor_Data.csv` (from the [Solar Power Generation Data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data) on Kaggle) is present in the root directory of this repository.

3.  **Configure R_HOME (if necessary):**
    The `rpy2` library may require the `R_HOME` environment variable to be set. The notebook contains a line `os.environ['R_HOME'] = '...'`. Please update this path to match your local R installation directory (e.g., `C:/Program Files/R/R-4.5.1` on Windows or `/usr/lib/R` on Linux).

4.  **Run the Jupyter Notebook:**
    Launch Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```
    Open `nonlinear.ipynb` and run all cells sequentially.

## 📊 Expected Output

Running the notebook will execute all analyses from the paper and save the resulting figures to the `img/` directory. The notebook cells correspond directly to the Listings in the SoftwareX paper:

* **Listing 1:** Generates **Figure 1** (U-shape example not captured by Pearson correlation).
* **Listing 2:** Generates **Figure 2** (Step-by-step visualization of the ACE algorithm).
* **Listing 3:** Runs the 16-scenario simulation, generating the data for **Table 3** (Simulation conditions) and **Table 4** (Correlation summary), as well as **Figure 3** (Method comparison bar chart).
* **Listing 4:** Calculates improvement ranks from the simulation, generating **Table 5** (Results ranked by dCor) and **Figures 6 & 7** (Improvement rank bar charts).
* **Listing 5:** Analyzes the real-world solar power data, generating **Table 6** (Solar data results) and **Figure 8** (Before/After ACE on solar data).

## 📄 Citation

If you use this code or the `acedcor` package in your research, please cite our paper:

```bibtex
@article{Park2025_SoftwareX,
  title = {acedcor: A Python package for detecting and diagnosing nonlinear relationships using ACE and Distance Correlation},
  author = {Sang Min Park and Hyoung-Moon Kim},
  journal = {SoftwareX},
  year = {2025},
  % ... (Add Volume, Pages, DOI when available)
}

@article{Park2025_ref,
  title = {Detecting Nonlinear Relationships Using Distance Correlation and Optimal Transformations},
  author = {Sang Min Park and Hyoung-Moon Kim},
  journal = {The Korean Journal of Applied Statistics},
  year = {2025},
  % ... (Add Volume, Pages, DOI when available)
}
