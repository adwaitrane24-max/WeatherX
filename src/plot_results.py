
import csv
from pathlib import Path

import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "performance_results.csv"
GRAPH_FILE = PROJECT_ROOT / "results" / "performance_graph.png"


def load_results():
    """Load performance measurements from the CSV file."""
    with open(RESULTS_FILE, "r", newline="") as file:
        reader = csv.DictReader(file)
        results = list(reader)

    if not results:
        raise ValueError(
            "No performance measurements found. "
            "Add the actual timing results to performance_results.csv first."
        )

    return results


def generate_graph():
    results = load_results()

    datasets = sorted(set(row["dataset"] for row in results))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for dataset in datasets:
        dataset_rows = [
            row for row in results if row["dataset"] == dataset
        ]

        dataset_rows.sort(key=lambda row: int(row["processes"]))

        processes = [int(row["processes"]) for row in dataset_rows]
        parallel_times = [
            float(row["parallel_time"]) for row in dataset_rows
        ]
        speedups = [
            float(row["speedup"]) for row in dataset_rows
        ]

        sequential_time = float(dataset_rows[0]["sequential_time"])

        # Graph 1: Execution time comparison
        axes[0].plot(
            processes,
            parallel_times,
            marker="o",
            label=f"{dataset} - MPI"
        )

        axes[0].axhline(
            y=sequential_time,
            linestyle="--",
            label=f"{dataset} - Sequential"
        )

        # Graph 2: Speedup
        axes[1].plot(
            processes,
            speedups,
            marker="o",
            label=dataset
        )

    axes[0].set_title("Sequential vs MPI Execution Time")
    axes[0].set_xlabel("Number of MPI Processes")
    axes[0].set_ylabel("Execution Time (seconds)")
    axes[0].set_xticks(sorted({
        int(row["processes"]) for row in results
    }))
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_title("MPI Speedup")
    axes[1].set_xlabel("Number of MPI Processes")
    axes[1].set_ylabel("Speedup")
    axes[1].set_xticks(sorted({
        int(row["processes"]) for row in results
    }))
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(GRAPH_FILE, dpi=300)
    plt.close()

    print(f"Performance graph saved at: {GRAPH_FILE}")


if __name__ == "__main__":
    generate_graph()
