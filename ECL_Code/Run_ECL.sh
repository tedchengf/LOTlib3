#!/bin/bash

args=''
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="${args} \"${i//\"/\\\"}\""
done

if [ "${args}" == "" ]; then args="/bin/bash"; fi

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

singularity \
    exec \
    --nv --overlay /scratch/fc1367/ECL_env/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif \
    /bin/bash -c "
unset -f which
source /opt/apps/lmod/lmod/init/sh
source /ext3/env.sh
conda activate base
${args}
"exit()