# Parallel Weather Data Analysis Using Python and MPI

## 1. Project Overview

This project implements a parallel weather data analysis system using Python and the Message Passing Interface (MPI).

The system analyses weather records containing temperature, humidity, rainfall, and location information. MPI distributes the dataset among multiple processes to perform computations in parallel and combine the results.

The project compares sequential and parallel execution to study the impact of parallel processing on performance.

## 2. Objectives

- Analyse large weather datasets efficiently.
- Implement parallel processing using Python and MPI.
- Distribute records among multiple processes using Scatter.
- Broadcast common parameters using Broadcast.
- Perform distributed calculations using Reduce and Allreduce.
- Collect individual process summaries using Gather.
- Compare sequential and parallel execution times.
- Evaluate speedup for different numbers of MPI processes.

## 3. Technologies Used

- Python
- MPI (Message Passing Interface)
- mpi4py
- Microsoft MPI
- CSV for dataset storage
- Matplotlib for performance visualisation
- VS Code

## 4. Project Structure

```text
WeatherX/
├── data/
│   ├── weather_data.csv
│   └── weather_data_100k.csv
├── docs/
├── src/
│   ├── __init__.py
│   ├── data_generator.py
│   ├── mpi_analysis.py
│   ├── sequential.py
│   ├── utils.py
│   └── plot_results.py
├── tests/
│   ├── __init__.py
│   └── test_results.py
├── results/
│   ├── performance_results.csv
│   └── performance_graph.png
└── README.md
```

## 5. MPI Implementation

The system uses the following MPI operations:

**MPI Initialization:** Initialises the MPI environment and identifies each process using its rank.

**Scatter:** Distributes the weather records among available processes.

**Broadcast:** Shares the extreme temperature threshold with all processes.

**Local Computation:** Each process calculates local temperature, humidity, rainfall, and extreme-temperature statistics.

**Gather:** Collects the local summaries from all processes at the root process.

**Reduce:** Combines local results to calculate global weather statistics.

**Allreduce:** Makes the combined temperature sum available to all processes.

## 6. Dataset Description

The project uses two CSV datasets:

| Dataset | Number of records |
|---|---:|
| weather_data.csv | 10,000 |
| weather_data_100k.csv | 100,000 |

The datasets contain the following attributes:

- `location_id`
- `temperature`
- `humidity`
- `rainfall`

## 7. How to Run the Project

### Install dependencies

```bash
python -m pip install mpi4py matplotlib
```

Microsoft MPI must also be installed on Windows.

### Run sequential analysis

```bash
python -m src.sequential
```

### Run parallel analysis with 4 processes (10K dataset)

```powershell
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 python src/mpi_analysis.py --input data/weather_data.csv
```

### Run parallel analysis with 4 processes (100K dataset)

```powershell
& "C:\Program Files\Microsoft MPI\Bin\mpiexec.exe" -n 4 python src/mpi_analysis.py --input data/weather_data_100k.csv
```

## 8. Testing and Validation

Unit testing was performed using Python's unittest framework.

The test cases validate:

- Total number of weather records.
- Average temperature.
- Minimum temperature.
- Maximum temperature.
- Average humidity.
- Total rainfall.
- Extreme temperature count.

**Testing result:** All 7 unit tests passed successfully.

The parallel implementation was also executed using 1, 2, and 4 MPI processes.

## 9. Performance Analysis

The execution times were measured for both datasets using different numbers of MPI processes.

| Dataset | MPI Processes | MPI Execution Time (seconds) |
|---|---:|---:|
| 10K | 1 | 0.036152 |
| 10K | 2 | 0.040674 |
| 10K | 4 | 0.045315 |
| 100K | 1 | 0.449964 |
| 100K | 2 | 0.441885 |
| 100K | 4 | 0.381661 |

Speedup is calculated using:

**Speedup = Sequential Execution Time / Parallel Execution Time**

The measured results indicate that increasing the number of processes does not always improve performance for smaller datasets because of MPI communication and coordination overhead.

For the 100K dataset, the four-process execution showed a lower execution time than the one-process execution.

The performance graph is available in `results/performance_graph.png`.

## 10. Conclusion

The project demonstrates the use of MPI-based parallel computing for weather data analysis. MPI operations such as Scatter, Broadcast, Gather, Reduce, and Allreduce enable data distribution, parallel computation, and result aggregation across multiple processes.

The experiments demonstrate that parallel processing can improve execution time for larger datasets, although communication overhead can affect performance for smaller datasets.

## 11. Future Scope

- Analyse larger weather datasets.
- Introduce additional weather parameters.
- Implement more advanced statistical analysis.
- Explore dynamic load balancing.
- Evaluate performance on distributed computing environments.
- Extend the system with weather prediction capabilities.

