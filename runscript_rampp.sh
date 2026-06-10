#!/usr/bin/env bash
#SBATCH -A NAISS2026-4-746
#SBATCH -p alvis
#SBATCH -o cluster_outputs/rampp_%j.out
#SBATCH --gpus-per-node=A40:1
#SBATCH -t 10:00:00

cd /cephyr/users/utsavp/Alvis/playground/actionrecognition

apptainer exec --nv \
    --overlay /mimer/NOBACKUP/groups/zijian/utsav/sam3_overlay.img \
    --bind /cephyr/users/utsavp/Alvis/playground:/workspace \
    /mimer/NOBACKUP/groups/zijian/utsav/lavila.sif \
    bash -c "\
        export HF_HOME=/mimer/NOBACKUP/groups/zijian/common; \
        export PYTHONPATH=/workspace; \
        source /opt/conda/etc/profile.d/conda.sh; \
        conda activate rampp; \
        cd /workspace/actionrecognition && bash run_rampp.sh \
    "



