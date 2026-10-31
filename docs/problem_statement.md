# Parallel Weather Data Analysis Using Python and MPI

*Sections written by Member 1: Introduction, Problem Statement, Objectives, Scope, Dataset Description*

## 1. Introduction

Weather stations collect large numbers of readings, such as temperature, humidity and rainfall. Summary statistics like the average temperature or the total rainfall have to be calculated from these readings. When the number of records is large, one program working through them one by one (sequential processing) takes longer as the data grows.

Distributed computing splits the work across several processes that run at the same time. This project uses MPI (Message Passing Interface) through the Python library `mpi4py`. The weather records are divided among processes, each process analyses its own share, and the partial results are combined into the final answer.

First, a sequential program produces correct reference results and a baseline execution time. The parallel MPI version is then checked against this baseline and its performance is compared with it.

## 2. Problem Statement

Given a dataset of weather records, where each record has a location ID, a temperature, a humidity and a rainfall value, calculate:

- the average, minimum and maximum temperature,
- the average humidity,
- the total rainfall,
- the number of records with temperature **strictly greater than 40 °C**.

Do this both sequentially and in parallel using MPI collective operations. Check that the two give the same results, and compare how long each takes.

## 3. Objectives

1. Generate a reproducible synthetic weather dataset of at least 10,000 records, scalable to 100,000.
2. Implement a sequential analysis program that serves as the correctness and timing baseline.
3. Distribute the records among MPI processes and compute the statistics in parallel.
4. Combine partial results using MPI collective operations (Scatter, Bcast, Reduce, Allreduce, Gather).
5. Verify that the parallel results match the sequential results.
6. Measure execution time and calculate speedup = sequential time / parallel time.

## 4. Scope

**In scope**
- Synthetic, locally generated CSV data (no external APIs or databases).
- Command-line Python programs.
- Testing with 1, 2 and 4 MPI processes on a single machine.
- Handling missing files, empty datasets and invalid CSV rows.

**Out of scope**
- Real-time or live weather feeds.
- Weather forecasting or prediction models.
- Web dashboards or graphical interfaces.
- Multi-machine cluster deployment (MPI supports it, but this project does not test it).

## 5. Dataset Description

The dataset is created by `src/data_generator.py` with a fixed random seed (`42`). Running the generator again therefore produces exactly the same file, which lets every team member work with identical data.

| Column | Type | Range | Meaning |
|---|---|---|---|
| `location_id` | integer | 1 – 100 | ID of the weather station |
| `temperature` | float (2 d.p.) | 5.00 – 48.00 °C | Air temperature |
| `humidity` | float (2 d.p.) | 10.00 – 100.00 % | Relative humidity |
| `rainfall` | float (2 d.p.) | 0.00 – 120.00 mm | Rainfall; about 60% of records are dry (0.0) |

The upper limit of the temperature range is above 40 °C on purpose, so that some records count as extreme.

**Files**

| File | Records | Command |
|---|---|---|
| `data/weather_data.csv` | 10,000 | `python -m src.data_generator` |
| `data/weather_data_100k.csv` | 100,000 | `python -m src.data_generator --records 100000 --output data/weather_data_100k.csv` |

**Sample rows** (first 5 rows of `data/weather_data.csv`, real generator output)

```
location_id,temperature,humidity,rainfall
82,9.79,76.74,0.0
18,36.67,70.9,10.52
55,6.37,18.43,0.0
78,6.14,27.9,65.44
29,24.32,35.04,91.08
```

**Validation rules applied when loading** (`src/sequential.py`, `parse_record`)

A row is skipped and counted as invalid if:
- any value is missing or is not a number,
- the temperature is outside −90 °C to 60 °C, or is NaN or infinity,
- the humidity is outside 0 – 100 %,
- the rainfall is negative,
- the `location_id` is less than 1.

**Sequential baseline results** (measured with `python -m src.sequential`)

| Statistic | 10,000 records | 100,000 records |
|---|---|---|
| Average temperature (°C) | 26.5890 | 26.5331 |
| Minimum temperature (°C) | 5.00 | 5.00 |
| Maximum temperature (°C) | 48.00 | 48.00 |
| Average humidity (%) | 54.8537 | 55.0395 |
| Total rainfall (mm) | 243284.99 | 2392991.81 |
| Records > 40 °C | 1834 | 18734 |

Timings change from run to run and from machine to machine. Member 4 records the final performance numbers. In one run on our machine, the computation step took about 0.0013 s for 10k records and 0.012 s for 100k records, and loading the CSV took most of the total time.
