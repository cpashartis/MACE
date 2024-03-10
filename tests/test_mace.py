import json
from unittest.mock import patch
from unittest import TestCase
import pytest
import os
import json
import tempfile
from scipy.spatial import KDTree
import numpy as np

import mace
from mace import MeshAdaptation


class TestMace(TestCase):
    def test_01_initialization(self):
        reference_points = np.random.rand(10, 3)
        optimizer = MeshAdaptation(reference_points, 5000, 0.9, 0.5, 500, 0.05)

        self.assertTrue(np.array_equal(optimizer.points, reference_points))
        self.assertEqual(optimizer.initial_temp, 5000)
        self.assertEqual(optimizer.alpha, 0.9)
        self.assertEqual(optimizer.min_temp, 0.5)
        self.assertEqual(optimizer.max_iterations, 500)
        self.assertEqual(optimizer.movement_scale, 0.05)
        self.assertIsInstance(optimizer.tree, KDTree)

    def test_02_objective_function(self):
        reference_points = np.array([[0, 0], [10, 10]])
        optimizer = MeshAdaptation(reference_points)
        current_points = np.array([[5, 5], [6, 6]])

        result = optimizer.objective_function(current_points)
        self.assertGreaterEqual(result, 0)

    @patch('numpy.random.choice')
    @patch('numpy.random.rand')
    def test_03_generate_new_configuration(self, mock_rand, mock_choice):
        mock_rand.return_value = np.array([0.5])
        mock_choice.return_value = np.array([0])

        reference_points = np.array([[0, 0], [10, 10]])
        current_configuration = np.array([[5, 5]])
        optimizer = MeshAdaptation(reference_points)

        new_configuration = optimizer.generate_new_configuration(current_configuration, 100, 1)
        self.assertEqual(new_configuration.shape, current_configuration.shape)

    def test_04_simulated_annealing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = os.path.join(temp_dir, "output.ndjson")

            reference_points = np.array([[0, 0], [10, 10]])
            initial_configuration = np.array([[5, 5], [6, 6]])
            optimizer = MeshAdaptation(reference_points)

            final_configuration = optimizer.simulated_annealing(initial_configuration, output_file)
            self.assertEqual(final_configuration.shape, initial_configuration.shape)
            self.assertTrue(os.path.exists(output_file))

        os.rmdir(temp_dir)

    def test_05_move_to_iterative_filename(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_file.txt")
            with open(file_path, "w") as f:
                f.write("content")

            MeshAdaptation.move_to_iterative_filename(file_path)

            self.assertFalse(os.path.exists(file_path))
            self.assertTrue(os.path.exists(file_path.replace(".txt", "_1.txt")))

        os.rmdir(temp_dir)

    @pytest.mark.integration
    def test_06_integration(self):
        data = np.loadtxt(os.path.join("data", "nial_fs.csv"), delimiter=',')
        points = data[:, :3]

        # Assuming `points` and `reference_points` are defined
        # Example usage:
        initial_temp = 10000
        alpha = 0.95
        min_temp = 1
        max_iterations = 1000

        optimizer = mace.MeshAdaptation(points, initial_temp, alpha, min_temp,
                                        max_iterations)

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_file.txt")
            optimizer.simulated_annealing(output_file=file_path)

        annealing_info = optimizer.load_configurations(file_path)
        self.assertIsNotNone(annealing_info)
        # ToDo: could use better tests here....

        os.rmdir(temp_dir)




