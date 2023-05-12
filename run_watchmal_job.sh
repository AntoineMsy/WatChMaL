#!/bin/bash
#SBATCH --job-name=jupyternotebook
#SBATCH --output=log-jupyter-%u-%J.txt
#SBATCH --error=log-jupyter-%u-%J.txt
#SBATCH --time=1-0:00:00
#SBATCH --mem=64G
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1

# Run WatChMaL in singularity after copying large files/directories to node's local disk

# usage: 
#        sbatch run_watchmal_job.sh -t -i singularity_image -c path_to_copy [-c another_path_to_copy] -w watchmal_directory -- watchmal_command [watchmal command options]
# -t                          Run in test mode (don’t copy files, run watchmal command with “-c job” option to print out the full config without actually running)
# -i singularity_image        Location of the singularity image to use
# -c path_to_copy             Copy file to node local storage for faster training
# -w watchmal_directory       Location of WatChMaL repository
# -- watchmal_command [opt]   Full command to run inside singularity is anything that comes after --
PATHS_TO_COPY=()
while [ $# -gt 0 ]; do
  case "$1" in
    -t)
      TEST=true
      ;;
    -i)
      shift
      SINGULARITY_FILE="$1"
      ;;
    -w)
      shift
      WATCHMAL_DIR="$1"
      ;;
    -c)
      shift
      PATHS_TO_COPY+=("$1")
      ;;
    -g)
      shift
      GPU_IDS="$1"
      ;;
    --)
      shift
      break
      ;;
  esac
  shift
done

if [ -z $WATCHMAL_DIR ]; then
  echo "WatChMaL directory not provided. Use -w option."
  exit 1;
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z $GPU_IDS ]; then
  echo "GPU ID(s) not specified. Use -g option."
  exit 1;
fi
export CUDA_VISIBLE_DEVICES=${GPU_IDS}

echo "entering directory $WATCHMAL_DIR"
cd "$WATCHMAL_DIR"

if [ -z $SINGULARITY_FILE ]; then
  echo "Singularity image file not provided. Use -i option."
  exit 1;
fi

export SINGULARITY_BIND="/home"

if [ -z $TEST ]; then
  for PATH_TO_COPY in "${PATHS_TO_COPY[@]}"; do
    echo "copying $PATH_TO_COPY to $SLURM_TMPDIR"
    #rsync -ahvPR "$PATH_TO_COPY" "$SLURM_TMPDIR"
    export SINGULARITY_BIND="${SINGULARITY_BIND},${SLURM_TMPDIR}/${PATH_TO_COPY##*/./}:${PATH_TO_COPY}"
  done
  SINGULARITY_FILE_MOVED="$SLURM_TMPDIR/${SINGULARITY_FILE##*/}"
  echo "copying singularity file from $SINGULARITY_FILE to $SINGULARITY_FILE_MOVED"
  #rsync -ahvP "$SINGULARITY_FILE" "$SINGULARITY_FILE_MOVED"
  echo "running command:"
  echo "  $@"
  echo "inside $SINGULARITY_FILE_MOVED"
  echo "with binds: $SINGULARITY_BIND"
  echo ""
  singularity exec --bind ${SINGULARITY_BIND} --nv "$SINGULARITY_FILE_MOVED" $@
else
  for PATH_TO_COPY in "${PATHS_TO_COPY[@]}"; do
    echo "skipping copying $PATH_TO_COPY to $SLURM_TMPDIR"
  done
  echo "running command:"
  echo "  $@ "
  echo "inside $SINGULARITY_FILE"
  echo ""
  singularity exec --nv --bind ${SINGULARITY_BIND} "$SINGULARITY_FILE" $@ # -c job
fi

