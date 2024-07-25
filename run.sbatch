#!/bin/bash
#SBATCH -J FS
#SBATCH --output=slurm-%j_%x.out                   
#SBATCH -N 1   
#SBATCH --nodes=1                
#SBATCH --ntasks-per-node=1     
#SBATCH --cpus-per-task=10   
#SBATCH --gres=gpu:1     


module load nvidia/cuda-11.1
module load nvidia/cuda-11.1_cudnn-8.2
echo "hostname :" 
hostname
echo "CUDA_VISIBLE_DEVICES = "$CUDA_VISIBLE_DEVICES
echo "CUDA_DEVICE_ORDER = "$CUDA_DEVICE_ORDER
set -e
set -x


dense="yes"
sparse="yes"

#----------------------- FeatureMethods
# QuickSelection - sum of strength
# output_attribution - attribution of output neurons
# Random - random importance


#----------------------- DST algorithm
# "DST_random"  ---> SET
# "DST"   ----> RigL


for seed in 0 1 2 3 4 5 6 7 8 9 
do
	for fs_method in "output_attribution" "QuickSelection" 
	do
		for density in 0.02 0.05 0.1 0.2 0.5 0.75  
		do
			for method in "DST_random"  "DST"  
			do 
				if [ "$sparse" = "yes" ]; then 
					python -m code.main_stand_sparse --seed $seed\
						--tag 'sparse_d='$density'_'$method'_'$fs_method\
						--prune-rate 0.3 --final-prune-epoch 80  \
						--dir_configs "configs_l2"\
						--fs_method $fs_method --method $method\
						--sparse-init "ERK" --model "MLP"\
						--final-density $density --init-density $density  \
						--update-frequency 10 --growth "gradient" --prune "magnitude" \
						--redistribution none --sparse \
						--exp "Experiment1_feature_selection"
			
				fi
			done
		done

		if [ "$dense" = "yes" ]; then 
			python -m code.main_stand_sparse --seed $seed --tag "dense_"$fs_method\
					--fs_method $fs_method  \
					--model "MLP" --dir_configs "configs_l2"\
					--exp "Experiment1_feature_selection"
		fi 
	done
done


