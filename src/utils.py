
"""
Project: Parallel Weather Data Analysis Using Python and MPI

Member 4:
Testing and Validation

This file compares sequential weather statistics
with the expected parallel MPI results.
"""

import sys
import math
import unittest
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequential import analyze_sequential


class TestWeatherAnalysis(unittest.TestCase):

    def setUp(self):
        self.csv_path = PROJECT_ROOT / "data" / "weather_data.csv"

        self.results = analyze_sequential(self.csv_path)

    def test_record_count(self):
        self.assertEqual(self.results["record_count"], 10000)

    def test_average_temperature(self):
        self.assertTrue(
            math.isclose(
                self.results["average_temperature"],
                26.588986,
                abs_tol=0.0001
            )
        )

    def test_minimum_temperature(self):
        self.assertEqual(
            self.results["min_temperature"], 5.0
        )

    def test_maximum_temperature(self):
        self.assertEqual(
            self.results["max_temperature"], 48.0
        )

    def test_average_humidity(self):
        self.assertTrue(
            math.isclose(
                self.results["average_humidity"],
                54.853747,
                abs_tol=0.0001
            )
        )

    def test_total_rainfall(self):
        self.assertTrue(
            math.isclose(
                self.results["total_rainfall"],
                243284.99,
                abs_tol=0.01
            )
        )

    def test_extreme_temperature_count(self):
        self.assertEqual(
            self.results["extreme_temperature_count"], 1834
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
