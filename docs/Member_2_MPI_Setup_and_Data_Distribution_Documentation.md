# Parallel Weather Data Analysis Using Python and MPI Collective Communication

## Member 2 — MPI Setup and Data Distribution

**Project:** Distributed Computing Mini-Project  
**Module:** MPI Initialization, CSV Loading, Data Partitioning, Scatter, and Broadcast  
**Primary source file:** `src/mpi_analysis.py`  
**Technologies:** Python, Open MPI, `mpi4py`, CSV

---

## 1. Objective

The objective of this module is to set up MPI communication and distribute weather observations among multiple Python processes for parallel analysis. It also ensures that every process receives the same extreme-temperature threshold.

This module performs the following tasks:

1. Initializes the MPI communicator and obtains each process's **rank** and the total **process count**.
2. Loads the weather CSV on the **root process (Rank 0)**.
3. Partitions the dataset into balanced chunks, including when the record count is not divisible by the number of processes.
4. Distributes records through **MPI Scatter** (`comm.scatter`).
5. Shares the extreme-temperature threshold (default **40°C**) through **MPI Broadcast** (`comm.bcast`).
6. Provides the distributed records and threshold to the parallel calculation module.
7. Verifies the distribution using **1, 2, and 4 processes**.

> **Scope:** Member 2 owns MPI setup and distribution. Local statistical calculations, reductions, gathering of summaries, and performance reporting are implemented by other team members as part of the integrated project.

## 2. System Architecture

The application follows an MPI **root–worker coordination model**. All processes execute the same program. Rank 0 loads the input file and prepares the chunks; all ranks participate in the collective communication operations and subsequently work on their own records.

```text
                     data/weather_data.csv
                             |
                             v
                    Rank 0: Read CSV
                             |
                             v
                    Balanced Partitioning
                             |
                             v
                        MPI Scatter
                             |
             +---------------+---------------+---------------+
             |               |               |               |
             v               v               v               v
          Rank 0          Rank 1          Rank 2          Rank 3
          Chunk 0         Chunk 1         Chunk 2         Chunk 3
             |               |               |               |
             +---------------+---------------+---------------+
                             |
          MPI Broadcast: threshold = 40°C to every rank
                             |
                             v
                 Local Weather Calculations
                    (Member 3 module)
                             |
                             v
                   Reduce / Allreduce
                    (Member 3 module)
                             |
                             v
                 Gather Process Summaries
                    (Member 4 module)
                             |
                             v
                    Final Weather Statistics
```

### Main MPI terminology

| Term | Meaning in this project |
|---|---|
| **Communicator** | `MPI.COMM_WORLD`, containing all launched processes. |
| **Rank** | Unique process identifier within the communicator, from `0` to `size - 1`. |
| **Size** | Number of participating MPI processes. |
| **Root** | Rank 0, which loads the data and supplies data/parameters to collectives. |
| **Collective operation** | Communication called by all ranks in the communicator in a compatible order. |

## 3. Environment and Execution

The development environment uses an MPI implementation (Open MPI) and the Python `mpi4py` library. The following commands are appropriate for Ubuntu/WSL2 Ubuntu:

```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev python3 python3-pip python3-venv

cd ~/weather-mpi-project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install mpi4py
```

Verify the environment:

```bash
mpiexec --version
python -c "from mpi4py import MPI; print(MPI.Get_version())"
```

Launch the module with four processes from the project root:

```bash
mpiexec -n 4 python src/mpi_analysis.py
```

The project can also be executed with `-n 1` or `-n 2` to test different process counts.

## 4. Input Dataset and Automatic Loading

The input is a CSV file located at:

```text
data/weather_data.csv
```

Required fields:

| Column | Purpose | Expected type |
|---|---|---|
| `location_id` | Station/region identifier | String |
| `temperature` | Temperature reading (°C) | Number |
| `humidity` | Relative humidity (%) | Number |
| `rainfall` | Rainfall reading | Number |

Example input (illustrative):

```csv
location_id,temperature,humidity,rainfall
Mumbai,32.0,70.0,1.2
Delhi,42.0,45.0,0.0
Pune,30.0,65.0,2.5
Jaipur,44.0,32.0,0.0
```

When the MPI program starts, **Rank 0 automatically reads the CSV from this path**. Users do not need to select or upload the dataset during each run. However, the program **does not download or generate a CSV automatically**: the file must already exist, typically prepared by Member 1. Replacing the file with another compatible dataset at the same path changes the input for the next execution.

The loading logic uses `csv.DictReader`, validates the required columns, and converts weather measurements to floating-point values.

```python
if rank == 0:
    records = load_weather_data(csv_path)
    chunks = divide_data(records, size)
else:
    chunks = None
```

The implementation also broadcasts any root-side loading error before reaching Scatter, avoiding a situation where worker processes wait for a collective operation that Rank 0 never enters.

## 5. MPI Initialization

Every launched process initializes its communication context using:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
```

For a four-process execution, `size = 4` on every rank, while `rank` takes a different value (`0`, `1`, `2`, or `3`) on each process. Rank 0 serves as the initial data-loading and distribution coordinator.

## 6. Balanced Data Partitioning

The root process divides the complete list of records into one chunk per MPI rank.

Let:

- **N** = total number of records;
- **P** = number of MPI processes;
- **B = N // P** = base chunk size;
- **R = N % P** = remaining records.

Each process receives `B` records, and the first `R` processes receive one additional record. This guarantees that no records are omitted or assigned twice during partitioning, and chunk sizes differ by at most one record.

For example, if **N = 11** and **P = 4**:

- `B = 11 // 4 = 2`
- `R = 11 % 4 = 3`
- Distribution: **3, 3, 3, 2** records.

| Rank | Records received (11-record example) |
|---|---:|
| 0 | 3 |
| 1 | 3 |
| 2 | 3 |
| 3 | 2 |
| **Total** | **11** |

Partitioning code:

```python
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
```

This also handles `N < P`: some ranks receive empty lists, which downstream calculations must handle safely.

## 7. MPI Scatter — Distributing Weather Records

**Scatter** distributes a different data chunk to each process. After Rank 0 creates a list containing exactly `P` chunks, every rank participates in:

```python
local_records = comm.scatter(chunks, root=0)
```

- **Rank 0:** Supplies the complete `chunks` list.
- **Other ranks:** Pass `None` as the scatter input and receive one chunk each.
- **All ranks:** Obtain `local_records`, a list of weather-record dictionaries.

The implementation uses lowercase `comm.scatter()`, which communicates Python objects and supports differently sized lists. It is distinct from the uppercase, buffer-based `comm.Scatter()` interface.

**Illustration (11 records / 4 ranks):**

```text
Root chunks -> [[R1,R2,R3], [R4,R5,R6], [R7,R8,R9], [R10,R11]]

Rank 0 <- [R1,R2,R3]
Rank 1 <- [R4,R5,R6]
Rank 2 <- [R7,R8,R9]
Rank 3 <- [R10,R11]
```

## 8. MPI Broadcast — Sharing the Temperature Threshold

**Broadcast** shares one common parameter with all ranks. The root defines the extreme-temperature threshold as **40°C**, and all processes execute:

```python
if rank == 0:
    extreme_threshold = 40.0
else:
    extreme_threshold = None

extreme_threshold = comm.bcast(extreme_threshold, root=0)
```

After `comm.bcast()`, every rank has `extreme_threshold = 40.0`, allowing Member 3's module to count records where:

```python
record["temperature"] > extreme_threshold
```

Using Broadcast avoids inconsistent per-rank settings and makes it easy to change the threshold centrally.

## 9. Distribution Algorithm

**Input:** CSV path, process count `P`, extreme-temperature threshold.  
**Output on each process:** `local_records`, `extreme_threshold`.

1. Launch `P` MPI processes.
2. Each process obtains its rank and communicator size.
3. Rank 0 loads and validates the CSV.
4. Rank 0 calculates balanced chunk sizes and builds a list of `P` chunks.
5. All processes participate in `comm.scatter()` and receive local records.
6. Rank 0 supplies the temperature threshold; all processes participate in `comm.bcast()`.
7. Each rank logs its record count and the received threshold for verification.
8. The local records and threshold become inputs to the parallel calculation stage.

> **MPI correctness requirement:** Collective operations must be entered by all processes in a compatible order. In the implementation, the loading-error broadcast occurs before Scatter, followed by the threshold broadcast. Team members integrating Reduce, Allreduce, or Gather must preserve a consistent collective-call sequence on all ranks.

## 10. Interface for Member 3

The distribution stage is exposed through:

```python
local_records, extreme_threshold = analyze_parallel(
    csv_path="data/weather_data.csv",
    threshold=40.0,
)
```

**Returned value 1 — `local_records`:** A list of dictionaries, for example:

```python
[
    {
        "location_id": "Mumbai",
        "temperature": 32.0,
        "humidity": 70.0,
        "rainfall": 1.2,
    }
]
```

**Returned value 2 — `extreme_threshold`:** A float (default `40.0`).

Member 3 can calculate local temperature sums, minima, maxima, humidity sums, rainfall sums, record counts, and extreme-temperature counts using these values. It then integrates `comm.reduce()` and `comm.allreduce()` into the full MPI workflow. Member 4 integrates `comm.gather()` and performance evaluation.

For the final integrated program, all ranks must proceed through the same collective communication sequence; avoid making any rank return or exit before a collective that other ranks still call.

## 11. Testing and Validation

The distribution module should be run with multiple process counts:

```bash
mpiexec -n 1 python src/mpi_analysis.py
mpiexec -n 2 python src/mpi_analysis.py
mpiexec -n 4 python src/mpi_analysis.py
```

### Illustrative distribution expectations

The following table uses **11 example records** to demonstrate the partitioning logic. These are **expected counts, not recorded execution measurements**.

| MPI processes | Expected records per rank | Expected total |
|---:|---|---:|
| 1 | `[11]` | 11 |
| 2 | `[6, 5]` | 11 |
| 4 | `[3, 3, 3, 2]` | 11 |

For the actual project dataset, capture and record the observed output:

| Run | Process count | Dataset rows | Observed rank-wise counts | Threshold verified | Result |
|---|---:|---:|---|---|---|
| 1 | 1 | *Fill from run* | *Fill from terminal* | *Yes/No* | *Pass/Fail* |
| 2 | 2 | *Fill from run* | *Fill from terminal* | *Yes/No* | *Pass/Fail* |
| 3 | 4 | *Fill from run* | *Fill from terminal* | *Yes/No* | *Pass/Fail* |

### Validation criteria

- All processes start and receive a unique rank.
- Only Rank 0 loads the full CSV before distribution.
- Every process receives the expected number of records.
- The sum of received record counts equals the dataset row count.
- No records are lost or duplicated during partitioning.
- The threshold on every process is identical to the root's setting.
- Uneven dataset sizes are handled correctly.
- Dataset-loading errors are reported without leaving workers blocked at Scatter.

**Evidence for final submission:** Attach screenshots of actual successful terminal runs with 1, 2, and 4 processes. Use the real dataset row count in the results table; do not use the illustrative figures as observed measurements.

## 12. Module Result and Conclusion

The Member 2 implementation establishes the MPI communication and distribution foundation for the weather analysis project. Rank 0 automatically reads the existing CSV input, divides its records into approximately equal-sized chunks, and distributes those chunks using `comm.scatter()`. Through `comm.bcast()`, all processes receive the same temperature threshold.

The module is designed to support uneven datasets and variable process counts. Its outputs provide the required inputs for Member 3's parallel calculations, while the full team integrates Reduce, Allreduce, Gather, and performance measurements to complete the project.

---

## References

1. Project team, *Parallel Weather Data Analysis Using Python and MPI — Work Division, Implementation Plan & Documentation Guide*, pp. 3–6.
2. mpi4py documentation: <https://mpi4py.readthedocs.io/>
3. Open MPI documentation: <https://docs.open-mpi.org/>
