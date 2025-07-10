
# LIMAO: A Framework for Lifelong Modular Learned Query Optimization
<p align="center">
    <img src="assets/LIMAO_icon.png" width="200"/>
<p>
This repository contains the implementation of LIMAO on Balsa, a Lifelong Modular Reinforcement Learning framework designed for database query optimization in dynamic environments. The codebase currently supports experiments on IMDB and TPC-H workloads, but you are welcome to extend it to additional workloads. Due to time constraints, the repository might still have some inconsistencies. Feel free to reach out to the authors for assistance!

Attention: 1. Please go to the corresponding branches and setup the environemnts, don't use the `main` branch directly. For exmaple, you can go to the `final_switching_workload` branch to setup the environment for LIMAO-Balsa's workload switch situation in IMDB; 2. Currently we have a several issues for the implementation: 1. We use zero masking instead of embedding decomposition to make the implementation easier; 2. Bao's current implementation of tree decomposition will cause some embedding issues, which will make the model instable for some queries ; 3. We haven't vectorized the modular training (assign the training data to different module combination, and train them in a batch mode) therefore the training could be costly in some GPUs that are not powerful enough. We will fix these issues soon; 4. We use the copied model to align with the Modular Lifelong Learning in robotics, however, this mechanism may not be necessary for query optimization, dropping this design may not impact the performance of LIMAO.

## Table of Contents
1. [Introduction](#introduction)
2. [System Requirements](#system-requirements)
3. [Setup](#setup)
4. [Datasets](#datasets)
5. [Usage](#usage)
6. [Customization](#customization)
7. [License](#license)

---

## Introduction

LIMAO integrates reinforcement learning to optimize database queries dynamically under workload, data volume, and both workload with data volume switching scenarios. This repository enables users to reproduce the experiments conducted in the associated paper.

## System Requirements

- **Operating System**: Linux (CloudLab c240g5 machine recommended, if you use your own mahcines, please make some customizations to the setup)
- **Python**: 3.8
- **PostgreSQL**: 12.5
- **Anaconda**: Anaconda3-2023.09

> **Note**: Additional configurations might be needed for using a different system.

## Setup

### 1. CloudLab Setup (Recommended)

1. **Prepare the storage environment**:
   ```bash
   sudo /usr/local/etc/emulab/mkextrafs.pl -f /mydata
   sudo chmod 777 /mydata
   ```
2. **Install Anaconda**:
   ```bash
   cd /mydata
   curl -O https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
   bash ./Anaconda3-2023.09-0-Linux-x86_64.sh
   rm ./Anaconda3-2023.09-0-Linux-x86_64.sh
   ```

3. **Clone the repository**:

   ```bash
   git clone git@github.com:Tsihan/LIMAOLifeLongRLDB.git -b final_switching_workload
   ```

> **Note**: This branch aims to replay the result in workload switching situation using LIMAO, if you want to see other dynamic situations, you may use other branches like final_switching_dbvolume, final_switching_workload_dbvolume, .etc. The prefix of each branch like "final", "original", "treedecompose" reflect different variants of the prototype evolving from Balsa to the final form of LIMAO implemented in Balsa.
4. **Set up the Python environment**:
   ```bash
   conda create -n limao python=3.8 -y
   conda activate limao
   pip install -e .
   pip install -e pg_executor
   pip install -r requirements.txt
   ```

5. **Install PostgreSQL 12.5**:
   ```bash
   wget https://ftp.postgresql.org/pub/source/v12.5/postgresql-12.5.tar.gz
   tar xzvf postgresql-12.5.tar.gz
   cd postgresql-12.5
   ./configure --prefix=/mydata/postgresql-12.5 --without-readline
   sudo make -j
   sudo make install
   echo 'export PATH=/mydata/postgresql-12.5/bin:$PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

6. **Install `pg_hint_plan` extension**:
   ```bash
   git clone https://github.com/ossc-db/pg_hint_plan.git -b REL12_1_3_7
   cd pg_hint_plan
   # Modify Makefile: Set PG_CONFIG to /mydata/postgresql-12.5/bin/pg_config
   make
   sudo make install
   ```

### 2. Prepare Datasets

1. **Download IMDB dataset**:
   ```bash
   mkdir -p /mydata/datasets/job && cd /mydata/datasets/job
   wget -c http://homepages.cwi.nl/~boncz/job/imdb.tgz
   tar -xvzf imdb.tgz
   ```

2. **Prepend headers to CSV files**:
   - Activate the LIMAO environment:
     ```bash
     conda activate limao
     ```
   - Run the script:
     ```bash
     python3 /mydata/LIMAOLifeLongRLDB/scripts/prepend_imdb_headers.py
     ```
   - Modify the script to set the correct directory:
     ```python
     flags.DEFINE_string('csv_dir', '/mydata/datasets/job', 'Directory to IMDB CSVs.')
     ```

3. **Initialize and start PostgreSQL**:
   ```bash
   pg_ctl -D /mydata/databases initdb
   cp /mydata/LIMAOLifeLongRLDB/conf/balsa-postgresql.conf /mydata/databases/postgresql.conf
   pg_ctl -D /mydata/databases start -l logfile
   ```

4. **Load IMDB dataset**:
   ```bash
   bash load-postgres/load_job_postgres.sh /mydata/datasets/job
   ```

> **Note**: For setting up the TPC-H workload, please refer to the following [Google Drive link](https://drive.google.com/drive/folders/1xoOIbTmW1IV6pr4QFruuKI6NGyiM3xTg?usp=drive_link).

## Usage

### 1. Run Experiments on IMDB Workloads

- **IMDB Set 1**:
  ```bash
  python3 run.py --run Balsa_JOBRandSplit_IMDB_assorted_3 --local
  ```
- **IMDB Set 2**:
  ```bash
  python3 run.py --run Balsa_JOBRandSplit_IMDB_assorted_4 --local
  ```
- **Switching Workload**:
  ```bash
  python3 run.py --run Balsa_JOBRandSplit_IMDB_assorted_Replay_2 --local
  ```

### 2. Run Experiments on TPCH Workloads

- **TPCH Set 1**:
  ```bash
  python3 run.py --run Balsa_JOBRandSplit_TPCH10_assorted --local
  ```
- **TPCH Set 2**:
  ```bash
  python3 run.py --run Balsa_JOBRandSplit_TPCH10_assorted_2 --local
  ```
- **Switching Workload**:
  ```bash
  python3 run.py --run Balsa_JOBRandSplit_TPCH10_assorted_Replay --local
  ```

## Customization

1. Modify parameters in the following scripts as needed:
   - `balsa/optimizer.py` (class initialization)
   - `balsa/util/postgres.py` (PostgreSQL connection)
   - `run.py`:
     - `_MakeWorkload(self, is_origin=False)`
     - `RunBaseline(self)`
   - `sim.py`:
     - `Params(cls)`
     - `_SimulationDataPath(self)`
     - `_FeaturizedDataPath(self)`

2. Update timezones in `balsa-postgresql.conf` if required.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

> **Note**: If any steps are unclear or additional details are required, please feel free to raise an issue in this repository.
