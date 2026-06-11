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






# for idx in range(0, n, 10):
#     point_x = test_df.iloc[idx]['x']
#     point_y = test_df.iloc[idx]['position']
#     center_position = (point_x, point_y)
    
#     point_cov = covar[test_df.index[idx] % len(covar)] 
    
#     ellipse = get_cov_ellipse(
#         cov=point_cov, 
#         pos=center_position, 
#         nstd=2, 
#         edgecolor='green', 
#         facecolor='green', 
#         alpha=0.2,
#         lw=1.5
#     )
    
#     ax.add_patch(ellipse)
#     ax.plot(point_x, point_y, 'o', color='green', markersize=4)