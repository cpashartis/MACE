import os
import numpy as np
import json

import pandas as pd
from scipy.spatial import KDTree


class MeshAdaptation:
    """
    A class for finding mesh configurations using simulated annealing and spatial queries given a mesh configuration,
    can also be used to find the mesh on a set of points with a thickness.

    Attributes:
        points (np.ndarray): An array of reference points in the mesh.
        initial_temp (float): The initial temperature for the simulated annealing process.
        alpha (float): The cooling rate of the simulated annealing process.
        min_temp (float): The minimum temperature to stop the simulated annealing process.
        max_iterations (int): The maximum number of iterations for the simulated annealing process.
        movement_scale (float): The scale of movement allowed in generating new configurations.
        tree (KDTree): A KD-tree built from the reference points for efficient spatial queries.
    """

    def __init__(self, points: np.ndarray, initial_temp: float = 10000, alpha: float = 0.95, min_temp: float = 1,
                 max_iterations: int = 1000, movement_scale: float = 0.1) -> None:
        """
        Initializes the MeshOptimizer with the given parameters.

        Args:
            points (np.ndarray): The reference points of the mesh.
            initial_temp (float): Initial temperature for the annealing process.
            alpha (float): Cooling rate for the annealing process.
            min_temp (float): Minimum temperature to stop the annealing process.
            max_iterations (int): Maximum iterations for the annealing process.
            movement_scale (float): Scale of movement for generating new configurations.
        """
        self.points = points
        self.tree = KDTree(self.points)  # Fast spatial queries
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.min_temp = min_temp
        self.max_iterations = max_iterations
        self.movement_scale = movement_scale

    def objective_function(self, current_points: np.ndarray,
                           nearest_counterpart_distance_factor: float = 0.05) -> float:
        """
        Calculates the objective function value for a given configuration of points.

        Args:
            current_points (np.ndarray): The current configuration of points.
            nearest_counterpart_distance_factor (float): Factor to determine the search radius for neighbors.

        Returns:
            float: The calculated objective function value.
        """
        lambda1 = 1
        obj_function = 0

        for current_point in current_points:
            distance, _ = self.tree.query(current_point, k=1)
            points_within_radius = self.tree.query_ball_point(current_point,
                                                              distance * (1 + nearest_counterpart_distance_factor))

            if points_within_radius:
                distances_within_radius = self.tree.query(current_point, k=len(points_within_radius))[0]
                distance_for_obj_function = np.mean(distances_within_radius)
            else:
                distance_for_obj_function = distance

            obj_function += lambda1 * distance_for_obj_function

        return obj_function

    def generate_new_configuration(self, current_configuration: np.ndarray, current_temp: float,
                                   min_temp: float) -> np.ndarray:
        """
        Generates a new configuration by moving a subset of points towards their nearest reference points.

        Args:
            current_configuration (np.ndarray): The current configuration of points.
            current_temp (float): The current temperature in the simulated annealing process.
            min_temp (float): The minimum temperature for the annealing process.

        Returns:
            np.ndarray: The new configuration of points.
        """
        num_points_to_move = int(len(current_configuration) * np.log10(current_temp / min_temp)) + 1
        num_points_to_move = min(num_points_to_move, len(current_configuration))
        indices_to_move = np.random.choice(len(current_configuration), num_points_to_move, replace=False)

        new_configuration = np.array(current_configuration, dtype=np.float64)

        for idx in indices_to_move:
            current_point = new_configuration[idx]
            _, nearest_idx = self.tree.query(current_point, k=1)
            nearest_reference_point = self.points[nearest_idx]
            movement_vector = nearest_reference_point - current_point
            scaled_movement_vector = movement_vector * self.movement_scale
            new_configuration[idx] += scaled_movement_vector

        return new_configuration

    def get_bounding_box_points(self, n_x=10, n_y=10, n_z=10):
        """
        Generates an initial configuration of points on the surface of a bounding box derived from the data.
        This box will then be contracted to the set of points you have initialized the class with.

        Args:
            data (np.ndarray): The reference data points to calculate the bounding box.
            n_x (int): Number of points along the x-axis on the box surface.
            n_y (int): Number of points along the y-axis on the box surface.
            n_z (int): Number of points along the z-axis on the box surface.

        Returns:
            np.ndarray: An array of points on the surface of the bounding box.
        """

        data = self.points
        # Get the outer bounds of the system
        x_min, x_max = np.min(data[:, 0]), np.max(data[:, 0])
        y_min, y_max = np.min(data[:, 1]), np.max(data[:, 1])
        z_min, z_max = np.min(data[:, 2]), np.max(data[:, 2])

        # Define box bounds and make slightly bigger than bounds
        box_bounds = np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])
        box_bounds[:, 0] -= 0.01
        box_bounds[:, 1] += 0.01

        # Generate points on the surface of the box
        x, y, z = np.linspace(*box_bounds[0, :], n_x), np.linspace(*box_bounds[1, :], n_y), np.linspace(
            *box_bounds[2, :], n_z)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

        return points

    def simulated_annealing(self, initial_configuration: np.ndarray = None,
                            output_file: str = 'sa_configurations.ndjson') -> np.ndarray:
        """
        Performs the simulated annealing process to optimize the mesh configuration.

        Args:
            initial_configuration (np.ndarray): The initial configuration of points to shrink to the reference
                points the class was initialized with. If None, the initial configuration is points on the surface
                of the bounding box.
            output_file (str): The path to the output file for logging configurations.

        Returns:
            np.ndarray: The optimized configuration of points.
        """
        current_configuration = initial_configuration if initial_configuration \
                                                         is not None else self.get_bounding_box_points()
        current_temp = self.initial_temp
        iteration = 0

        self.move_to_iterative_filename(output_file)

        while current_temp > self.min_temp and iteration < self.max_iterations:
            new_configuration = self.generate_new_configuration(current_configuration, current_temp, self.min_temp)
            delta_e = self.objective_function(new_configuration) - self.objective_function(current_configuration)

            if iteration % 10 == 0:
                print(iteration, delta_e)

            if delta_e < 0 or np.random.rand() < np.exp(-delta_e / current_temp):
                current_configuration = new_configuration

            with open(output_file, 'a') as f:
                config_data = {
                    'iteration': iteration,
                    'delta_e': delta_e,
                    'configuration': current_configuration.tolist(),
                    'temperature': current_temp
                }
                f.write(json.dumps(config_data) + '\n')

            current_temp *= self.alpha
            iteration += 1

        return current_configuration

    @staticmethod
    def move_to_iterative_filename(file_path: str) -> None:
        """
        Moves the given file to an iterative filename if it exists.

        Args:
            file_path (str): The path to the file to be moved.
        """
        if os.path.exists(file_path):
            base_name, extension = os.path.splitext(file_path)
            i = 1
            while True:
                iterative_file_path = f"{base_name}_{i}{extension}"
                if not os.path.exists(iterative_file_path):
                    os.rename(file_path, iterative_file_path)
                    print(f"File moved to: {iterative_file_path}")
                    break
                i += 1
        else:
            print("File does not exist.")

    @staticmethod
    def load_configurations(file_path='sa_configurations.ndjson') -> list:
        configurations = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                configuration = np.array(data['configuration'])
                configurations.append(configuration)
        return configurations
