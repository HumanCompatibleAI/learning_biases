#!/bin/bash
# This runs the training process for generating data
# The loops are the different parameters for the generated data
# This is the only place the parameters are recorded other than in the filenames

for seed in 32 1729 943 6543
do
	for imsize in 5 7 9 10 11 12
	do
		for rewardp in 0.0 0.3 0.5 0.7
		do
			for agent in 'optimal' 'naive' 'sophisticated' 'myopic'
			do
			  	echo "beginning: imsize-$imsize reward_prob-$rewardp agent-$agent seed-$seed"
			  	run_script="$(time python gridworld_data.py --imsize $imsize --reward_prob $rewardp --agent $agent --seed $seed)"
			  	echo "${run_script}"
			done
		done
	done
done
