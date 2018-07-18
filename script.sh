algos=(groupjoin)

sims=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)

datasets=(aol bms enron kosarak livejournal)

threads=(32 64)
scenarios=(1 2 3)

for dataset in ${datasets[@]}; do
  for algo in ${algos[@]}; do
    for sim in ${sims[@]}; do
      for thread in ${threads[@]}; do
         for scenario in ${scenarios[@]}; do
           Release/set_sim_join --threshold ${sim} --algorithm ${algo} --input /home/chribell/Desktop/datasets/final/${dataset}.txt --threads ${thread} --devmemory 5G --scenario ${scenario}
          done
      done
    done
  done
done
