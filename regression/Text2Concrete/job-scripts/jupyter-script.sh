#!/bin/env bash


ml purge

# Path to container
CONTAINER=/cephyr/users/andreeke/Alvis/LLMater/container-recipes/benchmarking-container.sif


apptainer exec --nv $CONTAINER jupyter notebook --config="${CONFIG_FILE}" 
