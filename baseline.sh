#!/bin/bash


for seed in 32 1729 943 6543
do
	for imsize in {5..20}
	do
		for wallp in $(seq 0.0 0.1 0.7)
		do
			for rewardp in $(seq 0.0 0.1 0.8)
			do
				for agent in 'optimal' 'naive' 'sophisticated' 'myopic'
				do
				  	echo "beginning: imsize-$imsize wall_prob-$wallp reward_prob-$rewardp agent-$agent seed-$seed"
				  	run_script="$(time python gridworld_data.py --imsize $imsize --wall_prob $wallp --reward_prob $rewardp --agent $agent --seed $seed)"
				  	echo "${run_script}"
				done
			done
		done
	done
done
