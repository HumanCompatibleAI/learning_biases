#!/bin/bash

for epochs in 30 60 90
do
	for seed in 32 1729 41 313 279 43
	do
		for imsize in {5..20}
		do
			for wallp in $(seq 0.0 0.1 0.7)
			do
				for rewardp in $(seq 0.0 0.1 0.8)
				do
					for agent in 'optimal' 'naive' 'sophisticated' 'myopic'
					do
					  rm -r /tmp/planner-vin/model/
					  echo "beginning: epoch-$epochs imsize-$imsize wall_prob-$wallp reward_prob-$rewardp agent-$agent seed-$seed"
					  run_script="$(time python train.py --imsize $imsize --epochs $epochs --wall_prob $wallp --reward_prob $rewardp --agent $agent --seed $seed> testing_files/imsize-$imsize-epochs-$epochs-wall_prob-$wallp-reward_prob-$rewardp-agent-$agent-seed-$seed.txt)"
					  mv predictioneval.png testing_files/imsize-$imsize-epochs-$epochs-wall_prob-$wallp-reward_prob-$rewardp-agent-$agent-seed-$seed.png
					  echo "${run_script}"
					done
				done
			done
		done
	done
done
