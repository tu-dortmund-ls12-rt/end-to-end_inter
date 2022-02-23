#!/usr/bin/env bash

nmb_processes=1 # number of concurrent processes

# Synthesis
python3 main.py --switch 1 --util 50 -n 10 --bench 'waters'
python3 main.py --switch 1 --util 50 -n 10 --bench 'uunifast'

python3 main.py --switch 2 --bench 'waters'
python3 main.py --switch 2 --bench 'uunifast'

# Single ECU experiments
python3 main.py --switch 3 --util 50 -n 10 --bench 'waters'
python3 main.py --switch 3 --util 50 -n 10 --bench 'uunifast'

# Inter ECU experiments
