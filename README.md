The repository is used to reproduce the evaluation from

*Compositional End-To-End Analysis under Execution time Variations and Mixture of Communication Means*

for RTAS 2022.

The output generated for the evaluation can be found here: https://tu-dortmund.sciebo.de/s/MkQSFTL8h48ZzHN


TO BE UPDATED:
# End-To-End Timing Analysis

The repository is used to reproduce the evaluation from

*Timing Analysis of Asynchronized Distributed Cause-Effect Chains*

for RTAS 2021.

This document is organized as follows:
1. [Environment Setup](#environment-setup)
2. [How to run the experiments](#how-to-run-the-experiments)
3. [Usage of Virtual Machine](#how-to-use-vm)
4. [Overview of the corresponding functions](#overview-of-the-corresponding-functions)
5. [Miscellaneous](#miscellaneous)

## Environment Setup
### Requirements

Some common software should be installed:
```
sudo apt-get install software-properties-common git screen python3.7
```
If the installation of Python3.7 doesn't work, likely you need to add deadsnakes PPA beforehand as it is not available on universe repo:
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
```

To run the experiments Python 3.7 is required (Python 3.9 might also work). Moreover, the following packages are required:
```
gc
argparse
math
numpy
scipy
random
matplotlib
operator
signal
```

Assuming that Python 3.7 is installed in the targeted machine, to install the required packages:
```
pip3 install scipy numpy matplotlib argparse
```
or
```
python3.7 -m pip install scipy numpy matplotlib argparse
```
In case there is any dependent package missing, please install them accordingly.

### File Structure

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

The experiments in the main function are splitted into 3 parts:
1. Single-ECU analysis
2. Interconnected ECU analysis
3. Plotting the results

In each step, the machines loads the results from the previous step, randomly creates necessary resources like task sets and cause-effect chains, and saves the results in the corresponsing folder in output.  

Besides that, there are two experiments for the runtime evaluation:
1. Dependency between number of jobs and runtime
2. Dependency between number of tasks and runtime for different hyperperiod maxima

The results of both experiments are stored in the folder output/runtime.

### Deployment

The following steps explain how to deploy this framework on the machine:

First, clone the git repository or download the [zip file](https://github.com/tu-dortmund-ls12-rt/end-to-end/archive/master.zip):
```
git clone https://github.com/tu-dortmund-ls12-rt/end-to-end.git
```
Move into the end-to-end folder, change the permissions of the script to be executable, and execute auto.sh natively:
```
cd end-to-end
chmod 777 auto.sh
./auto.sh 10
```
where 10 is maximal number of concurrent jobs on the machine, it can be set to a higher or lower number depending on the number of available cores.
Please note that the experiment may be very time consuming if used out of the box as explained in the section "How to run the experiments".
We encourage to use a machine with several cores or adjust the variables according to the section "How to use VM" to reduce the computation effort.

The runtime evaluation can be started independently from the same folder:
```
chmod 777 auto_runtime_jobs.sh
./auto_runtime_jobs.sh 10

chmod 777 auto_runtime_tasks.sh
./auto_runtime_tasks.sh 10
```
where again 10 is maximal number of concurrent jobs on the machine.

Please note that the parallelization will intensively use the computational resource and has to be adjusted for the number of cores of the machine.
```
'screen -ls' #shows all current screens
'killall screen' #aborts all current screens.
```

## How to run the experiments

- To ease the experiment deployment and parallelize the simulation on our server, we decide to use ```screen``` to manage persistent terminal sessions.
- The scripts ```auto.sh```, ```auto_runtime_jobs.sh``` and ```auto_runtime_tasks.sh``` are prepared for running all the experiments. They indicate the progress of the experiments by displaying short descriptions and timestamps.
- If the experiments have to be aborted at some time (e.g., because a certain package is missing), then the instructions inside the auto.sh file can be used to start step 2 and 3 of the evaluation manually.

- After finish of ```auto.sh```, the plots from Figure 7 and 8 of the paper can be found in the folder output/3plots:

Paper figure | Plot in output/3plots
--- | ---  
Fig. 7 (a) | davare_single_ecu_reaction_g=0.pdf
Fig. 7 (b) | davare_interconnected_reaction_g=0.pdf
Fig. 7 (c) | davare_single_ecu_age_g=0.pdf
Fig. 7 (d) | davare_interconnected_age_g=0.pdf
Fig. 8 (a) | davare_single_ecu_reaction_g=1.pdf
Fig. 8 (b) | davare_interconnected_reaction_g=1.pdf
Fig. 8 (c) | davare_single_ecu_age_g=1.pdf
Fig. 8 (d) | davare_interconnected_age_g=1.pdf

As a reference, we utilize a machine running Debian 4.19.98-1 (2020-01-26) x86_64 GNU/Linux, with 2 x AMD EPYC 7742 64-Core Processor (64 Cores, 128 Threads), i.e., in total 256 Threads with 2,25GHz and 256GB RAM. Running ```auto.sh 100``` to obtain the same plots from the paper takes about 1 hour with this machine.

- After finish of ```auto_runtime_jobs.sh``` and ```auto_runtime_tasks.sh```, the plots from Figure 9 and 10 of the paper can be found in the folder output/runtime:

Paper figure | Plot in output/runtime
--- | ---  
Fig. 9 | runtime_jobs.pdf
Fig. 10 | runtime_tasks.pdf

With the same machine as above, ```auto_runtime_jobs.sh 100``` is finished in less than one minute and ```auto_runtime_tasks.sh 100``` is finished after about 30 minutes.

## How to use VM

- Please download the [zip file](https://tu-dortmund.sciebo.de/s/GcftlevzCwg7Zaz), which contains the virtual disk and the machine description. The credential is: end2end/rtas21
- The source code is deployed on the desktop already. Some common software are installed accordingly, e.g., vim, vscode, git, evince.
- Please follow the above description to test out the provided analyses.
- Please note that the original scripts are for reproducing the results on the paper (so time consuming).
- The ```num_tries``` and ```runs_per_screen``` variables in ```auto.sh```, ```auto_runtime_jobs.sh``` and ```auto_runtime_tasks.sh``` can be reduced to obtain the results faster.

## Overview of the corresponding functions

The following tables describe the mapping between content of our paper and the source code in this repository.

**Section 5.B** (Local analysis):
Paper equation | Source code
--- | ---  
(12) Reaction time | utilities.analyzer.reaction_our()
(13) Data age | utilities.analyzer.max_age_our()

**Section 5.C** (Interconnected analysis):
Paper equation | Source code
--- | ---  
(17) Reaction time | utilities.analyzer.reaction_inter_our()
(18) Data age | utilities.analyzer.max_age_inter_our()

**Section 6** (Reduced data age):
Paper equation | Source code
--- | ---  
(23) Local | utilities.analyzer.max_age_our() with `reduced=True` Flag
(24) Interconnected | utilities.analyzer.max_age_inter_our() with `reduced=True` Flag

**Section 7** (Evaluation):
Benchmark | Source code
--- | ---  
\[6\] UUnifast benchmark for task sets | utilities.generator_UUNIFAST.gen_tasksets()
\[18\] Automotive benchmark for task sets | utilities.generator_WATERS.gen_tasksets()
\[18\] Automotive benchmark for cause-effect chains | utilities.generator_WATERS.gen_ce_chains()

Other analysis | Source code
--- | ---  
\[9\] Davare | utilities.analyzer.davare()
\[10\] Dürr: Data age | utilities.analyzer.age_duerr()
\[10\] Dürr: Reaction time | utilities.analyzer.reaction_duerr()
\[17\] Kloda | utilities.analyzer.kloda()

## Miscellaneous

### Authors

* Mario Günzel
* Marco Dürr
* Niklas Ueter
* Kuan-Hsun Chen
* Georg von der Brüggen
* Jian-Jia Chen

### Acknowledgments

This work has been supported by European Research Council (ERC) Consolidator Award 2019, as part of PropRT (Number 865170), and by Deutsche Forschungsgemeinschaft (DFG), as part of Sus-Aware (Project no. 398602212)

### License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
