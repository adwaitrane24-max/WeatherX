"""Synthetic weather dataset generator.

Owner: Member 1 (Dataset and Sequential Program)

This module creates a reproducible CSV file of fake (synthetic) weather
records. "Reproducible" means that running the generator again with the same
seed produces exactly the same file, so every team member works with
identical data and the sequential and parallel results can be compared.

CSV columns (shared with the sequential and MPI programs):
    location_id  - integer ID of the weather station (1 to num_locations)
    temperature  - degrees Celsius, 2 decimal places
    humidity     - relative humidity in percent (0 to 100), 2 decimal places
    rainfall     - rainfall in millimetres (0 or more), 2 decimal places

Usage from the project root:
    python -m src.data_generator
    python -m src.data_generator --records 100000 --output data/weather_data_100k.csv
    python -m src.data_generator --sample 10
"""

import argparse
import csv
import random
from pathlib import Path

# Project root = the folder that contains "src". Building paths from here
# means the script works no matter which folder it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "weather_data.csv"

# Column names used by every module in the project. Do not change the order
# or the spelling without telling the rest of the team.
CSV_COLUMNS = ["location_id", "temperature", "humidity", "rainfall"]

DEFAULT_NUM_RECORDS = 10_000
DEFAULT_SEED = 42
DEFAULT_NUM_LOCATIONS = 100

# Value ranges for the synthetic data. The temperature range goes above 40 C
# on purpose so that the "extreme temperature" count is not zero.
MIN_TEMPERATURE = 5.0
MAX_TEMPERATURE = 48.0
MIN_HUMIDITY = 10.0
MAX_HUMIDITY = 100.0
MAX_RAINFALL = 120.0
CHANCE_OF_DRY_DAY = 0.6  # 60% of records have no rain at all


def create_record(rng, num_locations):
    """Create one weather record as a dictionary.

    rng is a random.Random object. Using our own Random object (instead of the
    global random functions) keeps the sequence of numbers fully controlled by
    the seed.
    """
    location_id = rng.randint(1, num_locations)
    temperature = round(rng.uniform(MIN_TEMPERATURE, MAX_TEMPERATURE), 2)
    humidity = round(rng.uniform(MIN_HUMIDITY, MAX_HUMIDITY), 2)

    # Most days are dry, so rainfall is 0.0 most of the time.
    if rng.random() < CHANCE_OF_DRY_DAY:
        rainfall = 0.0
    else:
        rainfall = round(rng.uniform(0.1, MAX_RAINFALL), 2)

    return {
        "location_id": location_id,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
    }


def generate_records(num_records=DEFAULT_NUM_RECORDS, seed=DEFAULT_SEED,
                     num_locations=DEFAULT_NUM_LOCATIONS):
    """Return a list of num_records weather records (no file is written)."""
    if num_records < 0:
        raise ValueError("num_records cannot be negative")
    if num_locations < 1:
        raise ValueError("num_locations must be at least 1")

    # Same seed -> same random numbers -> same records every time.
    rng = random.Random(seed)
    return [create_record(rng, num_locations) for _ in range(num_records)]


def write_csv(records, output_path):
    """Write the records to a CSV file with a header row."""
    output_path = Path(output_path)
    # Create the "data" folder if it does not exist yet.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # newline="" stops Windows from inserting blank lines between rows.
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(records)


def generate_data(num_records=DEFAULT_NUM_RECORDS, output_path=DEFAULT_CSV_PATH,
                  seed=DEFAULT_SEED, num_locations=DEFAULT_NUM_LOCATIONS):
    """Generate the synthetic dataset and save it as a CSV file.

    This is the agreed team interface. Calling generate_data() with no
    arguments creates data/weather_data.csv with 10,000 records.

    Returns the path of the written CSV file.
    """
    records = generate_records(num_records, seed, num_locations)
    write_csv(records, output_path)
    return Path(output_path)


def print_sample_rows(csv_path, num_rows=5):
    """Print the first few rows of a CSV file (useful for documentation)."""
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        print(",".join(CSV_COLUMNS))
        for row_number, row in enumerate(reader):
            if row_number >= num_rows:
                break
            print(",".join(row[column] for column in CSV_COLUMNS))


def parse_arguments():
    """Read optional command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate a synthetic weather CSV dataset.")
    parser.add_argument("--records", type=int, default=DEFAULT_NUM_RECORDS,
                        help="number of records to generate (default: 10000)")
    parser.add_argument("--output", default=str(DEFAULT_CSV_PATH),
                        help="output CSV path (default: data/weather_data.csv)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed for reproducibility (default: 42)")
    parser.add_argument("--locations", type=int, default=DEFAULT_NUM_LOCATIONS,
                        help="number of distinct location IDs (default: 100)")
    parser.add_argument("--sample", type=int, default=5,
                        help="number of sample rows to print (default: 5)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    csv_path = generate_data(args.records, args.output, args.seed, args.locations)
    print(f"Generated {args.records} records (seed={args.seed}) -> {csv_path}")
    if args.sample > 0:
        print(f"\nFirst {args.sample} rows:")
        print_sample_rows(csv_path, args.sample)


# This block runs only when the file is executed directly,
# not when another module imports generate_data().
if __name__ == "__main__":
    main()
