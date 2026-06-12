import numpy as np
from matplotlib.patches import Ellipse

def get_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """
    Returns a matplotlib Ellipse patch given a 2x2 covariance matrix and a position.
    nstd: Number of standard deviations (e.g., 2 std devs covers ~95% of data).
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * nstd * np.sqrt(vals)
    
    return Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
