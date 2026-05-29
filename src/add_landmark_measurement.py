import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_landmark_measurement(graph, initial_estimate, result):
    # Determine the correct rotation (bearing) and distance from X(4) to L(2) 
    # rotation = 
    # distance = 
    x4 = initial_estimate.atPose2(X(4))
    l2 = result.atPoint2(L(2))

    # relative transform from pose to landmark

    # bearing (angle) and range (distance)
    dx = l2[0] - x4.x()
    dy = l2[1] - x4.y()

    distance = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - x4.theta()
    
    graph.add(gtsam.BearingRangeFactor2D(X(4), L(2), gtsam.Rot2.fromAngle(bearing), distance, MEASUREMENT_NOISE))
    return graph