"""
Project: Parallel Weather Data Analysis Using Python and MPI

Member 2:
MPI Setup, Data Distribution and Broadcasting
"""

from mpi4py import MPI
import csv
from pathlib import Path


# ----------------------------------------------------
# STEP 1: MPI INITIALIZATION
# ----------------------------------------------------

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()


# ----------------------------------------------------
# STEP 2: CSV DATA LOADING
# ----------------------------------------------------

def load_weather_data(file_path):

    records = []

    with open(file_path, "r", newline="") as file:

        reader = csv.DictReader(file)

        required_columns = {
            "location_id",
            "temperature",
            "humidity",
            "rainfall"
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "CSV must contain location_id, temperature, "
                "humidity, and rainfall columns."
            )

        for row in reader:

            record = {
                "location_id": row["location_id"],
                "temperature": float(row["temperature"]),
                "humidity": float(row["humidity"]),
                "rainfall": float(row["rainfall"])
            }

            records.append(record)

    return records


# ----------------------------------------------------
# STEP 3: BALANCED DATA PARTITIONING
# ----------------------------------------------------

def divide_data(records, num_processes):

    total_records = len(records)

    base = total_records // num_processes
    remainder = total_records % num_processes

    chunks = []

    start = 0

    for i in range(num_processes):

        count = base + (1 if i < remainder else 0)

        end = start + count

        chunks.append(records[start:end])

        start = end

    return chunks


# ----------------------------------------------------
# STEP 4: MPI DATA DISTRIBUTION
# ----------------------------------------------------

def analyze_parallel(
    csv_path="data/weather_data.csv",
    threshold=40.0
):

    # Root process loads the complete dataset.

    if rank == 0:

        try:
            records = load_weather_data(csv_path)

            print("\n========== MPI WEATHER ANALYSIS ==========")

            print(f"Total Processes: {size}")
            print(f"Total Records: {len(records)}")

            chunks = divide_data(records, size)

            error = None

        except (OSError, ValueError) as exc:

            chunks = None

            error = str(exc)

    else:

        chunks = None
        error = None


    # Notify all processes if loading failed.
    # This prevents other ranks from waiting indefinitely.

    error = comm.bcast(error, root=0)

    if error is not None:
        raise SystemExit(f"Dataset loading failed: {error}")


    # ------------------------------------------------
    # STEP 5: MPI SCATTER
    # ------------------------------------------------

    local_records = comm.scatter(chunks, root=0)


    # ------------------------------------------------
    # STEP 6: MPI BROADCAST
    # ------------------------------------------------

    if rank == 0:
        extreme_threshold = float(threshold)
    else:
        extreme_threshold = None

    extreme_threshold = comm.bcast(
        extreme_threshold,
        root=0
    )


    # ------------------------------------------------
    # STEP 7: VERIFY DATA DISTRIBUTION
    # ------------------------------------------------

    print(
        f"Rank {rank}: "
        f"Received {len(local_records)} records | "
        f"Threshold = {extreme_threshold}°C",
        flush=True
    )


    # Member 3 can continue local calculations here.

    return local_records, extreme_threshold


# ----------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    csv_file = project_root / "data" / "weather_data.csv"

    analyze_parallel(str(csv_file))
