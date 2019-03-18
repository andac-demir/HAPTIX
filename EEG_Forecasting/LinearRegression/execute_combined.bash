#!/bin/bash
#set a job name  
#SBATCH --job-name=Haptix System Identification 
#################  
#a file for job output, you can check job progress
#SBATCH --output=output_file.out
#################
# a file for errors from the job
#SBATCH --error=error_file.err
#################
#time you think you need; default is one day
#in minutes in this case, hh:mm:ss
#SBATCH --time=24:00:00
#################
#number of tasks you are requesting
#SBATCH -N 1
#SBATCH --exclusive
#################
#partition to use
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=120Gb
#################
#number of nodes to distribute n tasks across
#################

python lin_reg.py --all "true"