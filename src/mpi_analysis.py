
"""
Project: Parallel Weather Data Analysis Using Python and MPI

Member 2:
MPI Setup, Data Distribution and Broadcasting

Member 3:
Parallel Analysis and Reductions

Member 4:
Gather Integration and Performance Timing
"""

from mpi4py import MPI
import csv
import argparse
from pathlib import Path


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ----------------------------------------------------
# STEP 1: CSV DATA LOADING
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
# STEP 2: BALANCED DATA PARTITIONING
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
# STEP 3: PARALLEL WEATHER ANALYSIS
# ----------------------------------------------------

def analyze_parallel(csv_path, threshold=40.0):

    # Synchronize processes before starting the timer.
    comm.Barrier()
    start_time = MPI.Wtime()

    # Root process loads the dataset.
    if rank == 0:

        try:
            records = load_weather_data(csv_path)

            print("\n========== MPI WEATHER ANALYSIS ==========")
            print(f"Dataset: {Path(csv_path).name}")
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
    error = comm.bcast(error, root=0)

    if error is not None:
        raise SystemExit(f"Dataset loading failed: {error}")

    # ------------------------------------------------
    # STEP 4: MPI SCATTER
    # ------------------------------------------------

    local_records = comm.scatter(chunks, root=0)

    # ------------------------------------------------
    # STEP 5: MPI BROADCAST
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
    # STEP 6: VERIFY DATA DISTRIBUTION
    # ------------------------------------------------

    print(
        f"Rank {rank}: Received {len(local_records)} records | "
        f"Threshold = {extreme_threshold}°C",
        flush=True
    )

    # ------------------------------------------------
    # STEP 7: MEMBER 3 - LOCAL STATISTICS
    # ------------------------------------------------

    local_temperature_sum = sum(
        record["temperature"] for record in local_records
    )

    local_temperature_count = len(local_records)

    local_min_temperature = min(
        (record["temperature"] for record in local_records),
        default=float("inf")
    )

    local_max_temperature = max(
        (record["temperature"] for record in local_records),
        default=float("-inf")
    )

    local_humidity_sum = sum(
        record["humidity"] for record in local_records
    )

    local_rainfall_sum = sum(
        record["rainfall"] for record in local_records
    )

    local_extreme_count = sum(
        1 for record in local_records
        if record["temperature"] > extreme_threshold
    )

    print(f"\nRank {rank} Local Statistics:")
    print(f"Temperature Sum: {local_temperature_sum}")
    print(f"Temperature Count: {local_temperature_count}")
    print(f"Minimum Temperature: {local_min_temperature}")
    print(f"Maximum Temperature: {local_max_temperature}")
    print(f"Humidity Sum: {local_humidity_sum}")
    print(f"Rainfall Sum: {local_rainfall_sum}")
    print(f"Extreme Temperature Count: {local_extreme_count}")

    # ------------------------------------------------
    # STEP 8: MEMBER 4 - MPI GATHER
    # ------------------------------------------------

    local_summary = {
        "rank": rank,
        "records_processed": local_temperature_count,
        "temperature_sum": local_temperature_sum,
        "minimum_temperature": local_min_temperature,
        "maximum_temperature": local_max_temperature,
        "humidity_sum": local_humidity_sum,
        "rainfall_sum": local_rainfall_sum,
        "extreme_temperature_count": local_extreme_count
    }

    gathered_summaries = comm.gather(
        local_summary,
        root=0
    )

    if rank == 0:

        print("\n========== GATHERED PROCESS SUMMARIES ==========")

        for summary in gathered_summaries:

            print(f"\nProcess Rank: {summary['rank']}")
            print(f"Records Processed: {summary['records_processed']}")
            print(f"Temperature Sum: {summary['temperature_sum']}")
            print(f"Minimum Temperature: {summary['minimum_temperature']}")
            print(f"Maximum Temperature: {summary['maximum_temperature']}")
            print(f"Humidity Sum: {summary['humidity_sum']}")
            print(f"Rainfall Sum: {summary['rainfall_sum']}")
            print(
                "Extreme Temperature Count: "
                f"{summary['extreme_temperature_count']}"
            )

    # ------------------------------------------------
    # STEP 9: MEMBER 3 - MPI REDUCE
    # ------------------------------------------------

    global_temperature_sum = comm.reduce(
        local_temperature_sum, op=MPI.SUM, root=0
    )

    global_temperature_count = comm.reduce(
        local_temperature_count, op=MPI.SUM, root=0
    )

    global_min_temperature = comm.reduce(
        local_min_temperature, op=MPI.MIN, root=0
    )

    global_max_temperature = comm.reduce(
        local_max_temperature, op=MPI.MAX, root=0
    )

    global_humidity_sum = comm.reduce(
        local_humidity_sum, op=MPI.SUM, root=0
    )

    global_rainfall_sum = comm.reduce(
        local_rainfall_sum, op=MPI.SUM, root=0
    )

    global_extreme_count = comm.reduce(
        local_extreme_count, op=MPI.SUM, root=0
    )

    # ------------------------------------------------
    # STEP 10: MEMBER 3 - MPI ALLREDUCE
    # ------------------------------------------------

    all_temperature_sum = comm.allreduce(
        local_temperature_sum, op=MPI.SUM
    )

    # ------------------------------------------------
    # STEP 11: GLOBAL STATISTICS
    # ------------------------------------------------

    if rank == 0:

        if global_temperature_count > 0:

            average_temperature = (
                global_temperature_sum / global_temperature_count
            )

            average_humidity = (
                global_humidity_sum / global_temperature_count
            )

            display_min_temperature = global_min_temperature
            display_max_temperature = global_max_temperature

        else:
            average_temperature = None
            average_humidity = None
            display_min_temperature = None
            display_max_temperature = None

        print("\n========== GLOBAL WEATHER STATISTICS ==========")
        print(f"Total Records: {global_temperature_count}")
        print(f"Average Temperature: {average_temperature}")
        print(f"Minimum Temperature: {display_min_temperature}")
        print(f"Maximum Temperature: {display_max_temperature}")
        print(f"Average Humidity: {average_humidity}")
        print(f"Total Rainfall: {global_rainfall_sum}")
        print(f"Extreme Temperature Count: {global_extreme_count}")

    print(
        f"Rank {rank}: Global temperature sum using Allreduce = "
        f"{all_temperature_sum}",
        flush=True
    )

    # ------------------------------------------------
    # STEP 12: MPI EXECUTION TIME
    # ------------------------------------------------

    local_elapsed_time = MPI.Wtime() - start_time

    # Take the maximum elapsed time across processes.
    total_execution_time = comm.reduce(
        local_elapsed_time,
        op=MPI.MAX,
        root=0
    )

    if rank == 0:
        print(
            f"\nMPI_EXECUTION_TIME_SECONDS={total_execution_time:.6f}",
            flush=True
        )

    return local_records, extreme_threshold


# ----------------------------------------------------
# STEP 13: MAIN EXECUTION
# ----------------------------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Parallel Weather Data Analysis using MPI"
    )

    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "weather_data.csv"),
        help="Path to the weather CSV dataset"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=40.0,
        help="Extreme temperature threshold"
    )

    args = parser.parse_args()

    analyze_parallel(args.input, args.threshold)
