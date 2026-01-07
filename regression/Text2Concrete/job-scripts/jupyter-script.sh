#!/bin/env bash


ml purge

# Path to container
CONTAINER=/cephyr/users/andreeke/Alvis/LLMs-for-the-Design-of-Sustainable-Concretes/containers/benchmarking_container.sif


apptainer exec --nv $CONTAINER jupyter notebook --config="${CONFIG_FILE}" 
