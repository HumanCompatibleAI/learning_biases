#!/bin/bash

for epochs in 30 60 90
do
	for imsize in {5..20}
	do
		for wallp in $(seq 0.0 0.1 0.7)
		do
		  for rewardp in $(seq 0.0 0.1 0.8)
		  do
		  	for agent in ''
			  rm -r /tmp/planner-vin/model/
			  run_script="$(time python train.py -imsize $imsize -epochs $epochs -wall_prob $wallp -reward_prob $rewardp -agent $agent> testing_files/imsize-$imsize-epochs-$epochs-wall_prob-$wall_prob-reward_prob-$reward_prob-agent-$agent.txt)"
			  mv predictioneval.png imsize-$imsize-epochs-$epochs-wall_prob-$wallp-reward_prob-$rewardp-agent-$agent.png
			  echo "${run_script}" | grep "real" | awk '{print $2}'
		done
	done
done
