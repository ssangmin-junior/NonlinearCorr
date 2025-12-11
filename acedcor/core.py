import numpy as np
import pandas as pd
from scipy import stats
import warnings

# Try importing rpy2, handle failure gracefully for documentation generation or non-R environments
try:
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    R_AVAILABLE = True
except ImportError:
    R_AVAILABLE = False
    warnings.warn("rpy2 is not installed. acedcor functionality will be limited.")

def _check_r_dependencies():
    """
    Checks if required R packages (acepack, energy) are installed.
    """
    if not R_AVAILABLE:
        raise ImportError("rpy2 is required for this package.")
    
    try:
        importr('acepack')
        importr('energy')
    except Exception as e:
        raise ImportError(f"Required R packages not found: {e}. Please install 'acepack' and 'energy' in your R environment.")

def calculate_dcor_improvement(x, y, verbose=False):
    """
    Calculates the Distance Correlation (dCor) improvement after applying the ACE transformation.
    
    This function:
    1. Computes the baseline dCor between raw x and y.
    2. Applies the Alternating Conditional Expectation (ACE) algorithm to linearize the relationship.
    3. Computes the dCor between the transformed variables.
    4. Calculates the improvement (Delta dCor) and other diagnostic metrics.
    
    Parameters
    ----------
    x : array-like
        Independent variable (predictor). Should be 1D array.
    y : array-like
        Dependent variable (response). Should be 1D array.
    verbose : bool, optional
        If True, prints detailed steps. Default is False.
        
    Returns
    -------
    dict
        A dictionary containing:
        - 'dcor_before': Distance correlation of raw data.
        - 'dcor_after': Distance correlation after ACE transformation.
        - 'delta_dcor': Improvement (dCor_after - dCor_before).
        - 'pearson_before': Pearson correlation of raw data.
        - 'pearson_after': Pearson correlation after ACE transformation.
        - 'transformed_x': The ACE-transformed x array.
        - 'transformed_y': The ACE-transformed y array.
        
    Raises
    ------
    ImportError
        If rpy2 or R packages are missing.
    ValueError
        If input arrays are mismatched or empty.
    """
    
    _check_r_dependencies()
    
    # Ensure inputs are numpy arrays
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    if len(x) != len(y):
        raise ValueError("Input arrays x and y must have the same length.")
        
    # Python-side metrics (Before)
    pearson_before, _ = stats.pearsonr(x, y)
    
    # R context
    acepack = importr('acepack')
    energy = importr('energy')
    
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_x = ro.FloatVector(x)
        r_y = ro.FloatVector(y)
        
        # 1. dCor Before
        # energy.dcor returns a vector, index 0 is the dCor value
        dcor_before_r = energy.dcor(r_x, r_y)[0]
        
        # 2. ACE Transformation
        # acepack.ace returns a list-like object. 
        # typically: $tx (transformed x), $ty (transformed y)
        ace_res = acepack.ace(r_x, r_y)
        
        # Extract transformed vectors (R object to numpy)
        # Note: Depending on rpy2 version, access might vary. 
        # ace_res is a list vector. Indices: 0->tx, 1->ty usually.
        # It's safer to access by name if possible, but list index works for acepack.
        # acepack::ace returns list(tx, ty, ...)
        
        tx = np.array(ace_res[0]) # tx
        ty = np.array(ace_res[1]) # ty
        
        # 3. dCor After
        r_tx = ro.FloatVector(tx)
        r_ty = ro.FloatVector(ty)
        dcor_after_r = energy.dcor(r_tx, r_ty)[0]
        
    # Pearson After (on transformed data)
    pearson_after, _ = stats.pearsonr(tx, ty)
    
    results = {
        'dcor_before': float(dcor_before_r),
        'dcor_after': float(dcor_after_r),
        'delta_dcor': float(dcor_after_r - dcor_before_r),
        'pearson_before': float(pearson_before),
        'pearson_after': float(pearson_after),
        'transformed_x': tx,
        'transformed_y': ty
    }
    
    if verbose:
        print(f"Analysis Complete.")
        print(f"dCor Improvement: {results['dcor_before']:.4f} -> {results['dcor_after']:.4f} (+{results['delta_dcor']:.4f})")
        
    return results
