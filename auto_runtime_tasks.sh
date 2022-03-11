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

num_task_ind=19  # amount of different task numbers

hypers=( 0 1000 2000 3000 4000 )  # hyperperiods to be checked
len_hypers=${#hypers[@]}  # number of elements in hypers

timeout_par=0  # abort runtime measurement after ... seconds (0 = abort never)

for ((j=0;j<num_task_ind;j++))
do
  echo "task index $j"
  date
  for ((i=0;i<num_tries;i++))
  do
    for ((k=1;k<len_hypers;k++))
    do
      echo "start instance $(( (i)*(len_hypers-1)+k ))"
      screen -dmS ascr$j$i$k python3 runtime_tasks.py -n$(( (i)*(len_hypers-1)+k-1 )) -timeout=$timeout_par -tindex=$j -r$runs_per_screen -hypermin=$((${hypers[$((k-1))]})) -hypermax=$((${hypers[$k]}))

      # wait until variable is reached
      numberrec=$(screen -list | grep -c ascr.*)
      while (($numberrec >= $var))
      do
        sleep 1
        numberrec=$(screen -list | grep -c ascr.*)
      done
    done
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
python3 runtime_tasks.py -j1 -n=$(( num_tries*(len_hypers-1) ))


echo "DONE"
date
