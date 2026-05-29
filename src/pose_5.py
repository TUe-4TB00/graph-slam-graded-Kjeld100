import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

""" def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 


    # TODO: Perform the optimization and print the result

    return result """

def optimize(graph, initial_estimate):
    # Initialize optimizer
    params = gtsam.LevenbergMarquardtParams()
    
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Perform optimization
    result = optimizer.optimize()
    
    print("\nFinal Result:\n{}".format(result))

    return result

""" def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = "a"      # chosen pose option
    best_landmark = 1    # chosen landmark (1 or 2)
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
    marginals = []
    # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
    sum_of_marginals = 0
    return best_pose, best_landmark, sum_of_marginals """

def minimize_marginals(graph, initial_estimate, pose_options):
    # Making new maps to avoid modifying the original graph and initial estimate during testing
    work_graph = gtsam.NonlinearFactorGraph(graph)
    work_initial = gtsam.Values(initial_estimate)

    best_pose = "d"      # chosen pose option
    best_landmark = 1    # chosen landmark (1 or 2)
    
    pose_5 = pose_options[best_pose]
    work_graph, work_initial = add_pose(work_graph, work_initial, pose_5)
    
    result_step1 = optimize(work_graph, work_initial)
    work_graph = add_landmark_measurement(work_graph, result_step1, pose_5, best_landmark)
    final_result = optimize(work_graph, work_initial)

    marginals = gtsam.Marginals(work_graph, final_result)
    sum_of_marginals = marginals.marginalCovariance(L(1)).sum() + marginals.marginalCovariance(L(2)).sum()
    
    return best_pose, best_landmark, sum_of_marginals

""" def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = "a"      # chosen pose option
    best_landmark = 1    # chosen landmark (1 or 2)
    pose_5 = pose_options[best_pose]
    graph, initial_estimate = add_pose(graph, initial_estimate, pose_5)
    result = optimize(graph, initial_estimate)
    graph = add_landmark_measurement(graph, result, pose_5, best_landmark)
    result = optimize(graph, initial_estimate)

    # TODO: create a list of errors (each index corresponds to a pose) and add the error of each pose to the list
    list_of_errors = []
    # TODO: compute the sum of the errors and return it along with the best pose and landmark
    sum_of_errors = 0
    return best_pose, best_landmark, sum_of_errors  """

def minimize_errors(graph, initial_estimate, pose_options):
    best_score = float("inf")

    for pose_key, pose_5 in pose_options.items():

        temp_graph = gtsam.NonlinearFactorGraph(graph)
        temp_values = gtsam.Values(initial_estimate)

        temp_graph, temp_values = add_pose(temp_graph, temp_values, pose_5)

        result = optimize(temp_graph, temp_values)

        for landmark in [1, 2]:
            temp_graph2 = gtsam.NonlinearFactorGraph(temp_graph)
            temp_values2 = gtsam.Values(temp_values)

            temp_graph2 = add_landmark_measurement(
                temp_graph2, result, pose_5, landmark
            )

            final_result = optimize(temp_graph2, temp_values2)

            marginals = gtsam.Marginals(temp_graph2, final_result)
            selection_metric = sum(
                marginals.marginalCovariance(X(i)).trace()
                for i in [1, 2, 3]
            )

            returned_metric = sum(
                np.sum(np.abs(result.atPose2(key).matrix() - true_pose.matrix()))
                for key, true_pose in [
                    (X(1), gtsam.Pose2(0.0, 0.0, 0.0)),
                    (X(2), gtsam.Pose2(2.0, 0.0, 0.0)),
                    (X(3), gtsam.Pose2(4.0, 0.0, 0.0)),
                ]
            )

            if selection_metric < best_score:
                best_score = selection_metric
                best_pose = pose_key
                best_landmark = landmark
                best_returned_score = returned_metric

    return best_pose, best_landmark, best_returned_score