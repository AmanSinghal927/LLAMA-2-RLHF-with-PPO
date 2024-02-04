# Chosing r_rejected

![alt text](chosing_r_rejected.png)

# Launching Code on HPC
```
#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --gres=gpu:v100:4
#SBATCH --nodes 1
#SBATCH --account=csci_ga_3033_102-2023fa
#SBATCH --partition=n1c24m128-v100-4
#SBATCH --time=20:10:00
#SBATCH --mail-type=END
#SBATCH --output=%jmain.out
#SBATCH --error=%jmaintester.err
module purge
singularity exec --nv --bind /scratch/as14661 --overlay /scratch/as14661/as14661/jup_env/my_pytorch.ext3:ro /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash \
-c "source /ext3/env.sh; cd /scratch/as14661/as14661/trl/examples/scripts; python reward_modeling.py"
```
