"""Sequential (single-process) weather analysis.

Owner: Member 1 (Dataset and Sequential Program)

This is the baseline program. It reads the whole CSV file in one process and
calculates the weather statistics one record at a time. It does NOT use MPI.
The MPI program must produce the same statistics, and its execution time is
compared against the time measured here to calculate speedup.

Statistics calculated:
    - average, minimum and maximum temperature
    - average humidity
    - total rainfall
    - number of records with temperature strictly greater than 40 C

Usage from the project root:
    python -m src.sequential
    python -m src.sequential --input data/weather_data_100k.csv
"""

import argparse
import csv
import math
import time
from pathlib import Path

from src.data_generator import CSV_COLUMNS, DEFAULT_CSV_PATH

# Shared with the MPI program so both use exactly the same rule.
# A record is "extreme" only if temperature > 40.0 (40.0 itself is NOT extreme).
EXTREME_TEMP_THRESHOLD = 40.0

# Limits used to reject impossible values when reading the CSV.
VALID_TEMPERATURE_RANGE = (-90.0, 60.0)  # roughly the Earth's recorded extremes
VALID_HUMIDITY_RANGE = (0.0, 100.0)


def parse_record(row):
    """Convert one CSV row (a dict of strings) into a clean record.

    Returns a dict with proper number types, or None if the row is invalid
    (missing values, text that is not a number, or values outside valid ranges).
    """
    try:
        location_id = int(row["location_id"])
        temperature = float(row["temperature"])
        humidity = float(row["humidity"])
        rainfall = float(row["rainfall"])
    except (KeyError, TypeError, ValueError):
        # KeyError: column missing, TypeError: value is None (short row),
        # ValueError: text such as "abc" or "" that cannot become a number.
        return None

    # float("nan") and float("inf") parse successfully, so reject them here.
    if not all(math.isfinite(value) for value in (temperature, humidity, rainfall)):
        return None
    if location_id < 1:
        return None
    if not VALID_TEMPERATURE_RANGE[0] <= temperature <= VALID_TEMPERATURE_RANGE[1]:
        return None
    if not VALID_HUMIDITY_RANGE[0] <= humidity <= VALID_HUMIDITY_RANGE[1]:
        return None
    if rainfall < 0:
        return None

    return {
        "location_id": location_id,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
    }


def load_weather_data(csv_path=DEFAULT_CSV_PATH):
    """Read the CSV file and return (valid_records, invalid_record_count).

    Invalid rows are skipped and counted instead of crashing the program.
    Raises FileNotFoundError with a helpful message if the file is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Generate it first with:  python -m src.data_generator"
        )

    valid_records = []
    invalid_record_count = 0

    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        # A completely empty file has no header row at all.
        if reader.fieldnames is None:
            return valid_records, invalid_record_count

        missing_columns = [c for c in CSV_COLUMNS if c not in reader.fieldnames]
        if missing_columns:
            raise ValueError(f"CSV file is missing required columns: {missing_columns}")

        for row in reader:
            record = parse_record(row)
            if record is None:
                invalid_record_count += 1
            else:
                valid_records.append(record)

    return valid_records, invalid_record_count


def compute_statistics(records, threshold=EXTREME_TEMP_THRESHOLD):
    """Calculate all weather statistics for a list of records.

    Uses a single loop over the data and keeps running totals, which is the
    same idea the parallel version uses on each process (sum, count, min, max).
    """
    record_count = len(records)

    # Empty dataset: there is nothing to average, so return None for values
    # that cannot be calculated instead of dividing by zero.
    if record_count == 0:
        return {
            "record_count": 0,
            "average_temperature": None,
            "min_temperature": None,
            "max_temperature": None,
            "average_humidity": None,
            "total_rainfall": 0.0,
            "extreme_temperature_count": 0,
        }

    temperature_sum = 0.0
    humidity_sum = 0.0
    rainfall_sum = 0.0
    extreme_count = 0
    # Start min/max at infinity so the first record always replaces them.
    min_temperature = math.inf
    max_temperature = -math.inf

    for record in records:
        temperature = record["temperature"]
        temperature_sum += temperature
        humidity_sum += record["humidity"]
        rainfall_sum += record["rainfall"]

        if temperature < min_temperature:
            min_temperature = temperature
        if temperature > max_temperature:
            max_temperature = temperature

        # Strictly greater than: exactly 40.0 is not counted.
        if temperature > threshold:
            extreme_count += 1

    # Average = total sum / total count
    return {
        "record_count": record_count,
        "average_temperature": temperature_sum / record_count,
        "min_temperature": min_temperature,
        "max_temperature": max_temperature,
        "average_humidity": humidity_sum / record_count,
        "total_rainfall": rainfall_sum,
        "extreme_temperature_count": extreme_count,
    }


def analyze_sequential(csv_path=DEFAULT_CSV_PATH, threshold=EXTREME_TEMP_THRESHOLD):
    """Load the dataset, calculate statistics and measure execution time.

    This is the agreed team interface. Returns a dictionary containing the
    statistics plus:
        invalid_record_count     - rows skipped because they were invalid
        threshold                - the extreme-temperature threshold used
        load_time_seconds        - time spent reading the CSV file
        execution_time_seconds   - time spent calculating the statistics
        total_time_seconds       - load time + execution time
    """
    # time.perf_counter() is a high-resolution clock meant for measuring
    # short durations. We take the difference between two readings.
    load_start = time.perf_counter()
    records, invalid_record_count = load_weather_data(csv_path)
    load_end = time.perf_counter()

    compute_start = time.perf_counter()
    results = compute_statistics(records, threshold)
    compute_end = time.perf_counter()

    results["invalid_record_count"] = invalid_record_count
    results["threshold"] = threshold
    results["load_time_seconds"] = load_end - load_start
    results["execution_time_seconds"] = compute_end - compute_start
    results["total_time_seconds"] = compute_end - load_start
    return results


def format_value(value, decimals=2):
    """Format a number for printing; show 'N/A' when it is None."""
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def print_results(results, csv_path):
    """Print the results in a readable table."""
    print("=" * 50)
    print(" SEQUENTIAL WEATHER ANALYSIS")
    print("=" * 50)
    print(f"Dataset                  : {csv_path}")
    print(f"Valid records            : {results['record_count']}")
    print(f"Invalid records skipped  : {results['invalid_record_count']}")
    print("-" * 50)
    print(f"Average temperature (C)  : {format_value(results['average_temperature'], 4)}")
    print(f"Minimum temperature (C)  : {format_value(results['min_temperature'])}")
    print(f"Maximum temperature (C)  : {format_value(results['max_temperature'])}")
    print(f"Average humidity (%)     : {format_value(results['average_humidity'], 4)}")
    print(f"Total rainfall (mm)      : {format_value(results['total_rainfall'])}")
    label = f"Records > {results['threshold']} C"
    print(f"{label:<25}: {results['extreme_temperature_count']}")
    print("-" * 50)
    print(f"CSV load time (s)        : {results['load_time_seconds']:.6f}")
    print(f"Computation time (s)     : {results['execution_time_seconds']:.6f}")
    print(f"Total time (s)           : {results['total_time_seconds']:.6f}")
    print("=" * 50)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the sequential weather analysis.")
    parser.add_argument("--input", default=str(DEFAULT_CSV_PATH),
                        help="CSV dataset path (default: data/weather_data.csv)")
    parser.add_argument("--threshold", type=float, default=EXTREME_TEMP_THRESHOLD,
                        help="extreme temperature threshold in C (default: 40.0)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    try:
        results = analyze_sequential(args.input, args.threshold)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")
        raise SystemExit(1)
    print_results(results, args.input)


if __name__ == "__main__":
    main()
