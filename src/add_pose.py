
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    odometry = gtsam.Pose2(2.0, 0.0, np.pi / 2)  # Example odometry measurement (dx, dy, dtheta)
    
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), odometry, ODOMETRY_NOISE))
    
    est_pose_4 = gtsam.Pose2(4+np.sqrt(2), np.sqrt(2), np.pi / 2)
    
    if initial_estimate.exists(X(4)):
        initial_estimate.update(X(4), est_pose_4)
    else:
        initial_estimate.insert(X(4), est_pose_4)
    
    return graph, initial_estimate