#!/usr/bin/env bash

nmb_processes="$1" # number of concurrent processes
utils="50 60 70 80 90"
nmb_tasksets=1000
nmb_interchains=10000


# Single ECU Synthesis
#for ut in $utils
#do
#  python3 main.py --switch=1 --util=$ut --number=$nmb_tasksets --bench='waters' --proc=$nmb_processes
#  python3 main.py --switch=1 --util=$ut --number=$nmb_tasksets --bench='uunifast' --proc=$nmb_processes
#done
#
## Single ECU experiments
#for ut in $utils
#do
#  python3 main.py --switch=2 --util=$ut --number=$nmb_tasksets --bench='waters' --proc=$nmb_processes
#  python3 main.py --switch=2 --util=$ut --number=$nmb_tasksets --bench='uunifast' --proc=$nmb_processes
#done
#
## Inter ECU Synthesis
#python3 main.py --switch=3 --number=$nmb_interchains --bench='waters'
#python3 main.py --switch=3 --number=$nmb_interchains --bench='uunifast'
#
## Inter ECU experiments
#python3 main.py --switch=4 --bench='waters' --proc=$nmb_processes
#python3 main.py --switch=4 --bench='uunifast' --proc=$nmb_processes

# Single ECU plotting
python3 main.py --switch=5 --bench='waters'
python3 main.py --switch=5 --bench='uunifast'

# Inter ECU plotting
python3 main.py --switch=6 --bench='waters'
python3 main.py --switch=6 --bench='uunifast'
