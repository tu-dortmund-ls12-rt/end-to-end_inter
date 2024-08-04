# Compositional End-To-End Timing Analysis

The repository is used to reproduce the evaluation from

*Compositional Timing Analysis of Asynchronized Distributed Cause-effect Chains*

published at IEEE TECS 2023.
This is a Journal Version of the paper: 
*Timing analysis of asynchronized distributed cause-effect chains*
presented at RTAS 2021.

Specifically, in this version, the plots of the dissertation of Mario Günzel are generation.

## Environment Setup

### STEP1: Install common softward

This evaluation uses Python3.7 (newer versions might work as well) and screen.

To install common software, use:

```
sudo apt-get install software-properties-common git screen python3.7
```

If the installation of Python3.7 doesn't work, likely you need to add deadsnakes PPA beforehand as it is not available
on universe repo:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

### STEP2: Clone Github Script

This can be done by using 

```
git clone <RepoURL>
```

### STEP3: Start Python Venv and Install Requirements

Start a python venv using 
```
python3.7 -m venv venv 
source venv/bin/activate
```

Load requirements:

```
pip install -r requirements.txt
```


## Execute Experiments

Make sure that the virtual environment is loaded (every time before a script is run).
```
source venv/bin/activate
```
Afterwards, execute the scripts ```auto.sh```, ```auto_runtime_jobs.sh``` and ```auto_runtime_tasks.sh``` to run the experiments:
```
./auto.sh 100
./auto_runtime_jobs.sh 100
./auto_runtime_tasks.sh 100
```
where ```100``` indicates the number of currently running jobs on the machine. 
If the generation process was not successfull, the scripts may have to be rerun, until all data was generated successfully. 
The generated plots can be found in 
```output/3plots```
for the latency reduction and gap reduction, and in 
```output/runtime```
for the runtime evaluations.

As a reference, we utilize a machine running Debian 4.19.98-1 (2020-01-26) x86_64 GNU/Linux, with 2 x AMD EPYC 7742
64-Core Processor (64 Cores, 128 Threads), i.e., in total 256 Threads with 2,25GHz and 256GB RAM.
Running ```auto.sh 100``` takes about 1 hour with this machine.
Furthermore, ```auto_runtime_jobs.sh 100``` is finished in less than one minute
and ```auto_runtime_tasks.sh 100``` is finished after about 30 minutes.

The scripts use ```screen``` to run several instances concurrently. 
If the script is stopped for some reason, the still open screen instances should be checked and killed if necessary:
```
'screen -ls' #shows all current screens
'killall screen' #aborts all current screens.
```


## File Structure

    .
    ├── output                       # Placeholder for outputs
    │   ├── 1single                  # Single ECU chains + results
    │   ├── 2interconn               # Interconnected ECU chains + result
    |   ├── 3plots                   # Plots as in the paper
    │   └── runtime                  # Output for the runtime evaluation
    ├── utilities                    # Placeholder for additional files
    │   ├── analyzer.py              # Methods to analyze end-to-end timing behavior
    │   ├── augmented_job_chain.py   # Augmented job chains as in the paper
    │   ├── chain.py                 # Cause-effect chains
    │   ├── communication.py         # Communication tasks
    │   ├── evaluation.py            # Methods to draw plots
    │   ├── event_simulator.py       # Event-driven simulator with fixed execution time
    │   ├── generator_UUNIFAST       # Task set generator for uunifast benchmark
    │   ├── generator_WATERS         # Task set and cause-effect chain generator for waters benchmark
    │   ├── task.py                  # Tasks
    │   └── transformer.py           # Connect task creating with the scheduler
    ├── auto.sh                      # Running all experiments automatically
    ├── main.py                      # Main function
    ├── auto_runtime_jobs.sh         # Running first runtime evaluation automatically
    ├── runtime_jobs.py              # First runtime evaluation
    ├── auto_runtime_tasks.sh        # Running second runtime evaluation automatically
    ├── runtime_tasks.py             # Second runtime evaluation
    └── README.md

## Miscellaneous

### Authors

* Mario Günzel
* Marco Dürr
* Niklas Ueter
* Kuan-Hsun Chen
* Georg von der Brüggen
* Jian-Jia Chen

### Acknowledgments

This work has been supported by European Research Council (ERC) Consolidator Award 2019, as part of PropRT (Number
865170), and by Deutsche Forschungsgemeinschaft (DFG), as part of Sus-Aware (Project no. 398602212)

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
