#!/bin/bash

###
# Specify number of concurrent instances.
###
if [ $# -eq 0 ]
then
  echo "Specify maximal number of concurrent instances for the experiment (e.g. './auto.sh 5' )."
  exit 1
else
  var=$1
  echo "with $var concurrent instances"
fi


###
# Timing.
###
echo "===Run Experiment"
date

num_tries=100  # number of runs
runs_per_screen=10  # number of runs per screen

for ((i=0;i<num_tries;i++))
do
  echo "start instance $i"
  screen -dmS ascr$i python3 runtime_jobs.py -r$runs_per_screen -n$i -jobmax=100000

  # wait until variable is reached
  numberrec=$(screen -list | grep -c ascr.*)
  while (($numberrec >= $var))
  do
    sleep 1
    numberrec=$(screen -list | grep -c ascr.*)
  done
done

# wait until all are closed
while screen -list | grep -q ascr.*
do
  sleep 1
done

###
# Plotting.
###
echo "===Plot data"
date
python3 runtime_jobs.py -j1 -n$num_tries


echo "DONE"
date
