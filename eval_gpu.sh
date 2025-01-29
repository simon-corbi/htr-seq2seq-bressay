#!/bin/bash
#SBATCH -o /proj/document_analysis/users/x_scorb/logs/multiscripts/%j.out
#SBATCH -e /proj/document_analysis/users/x_scorb/logs/multiscripts/%j.err
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 4                           # one CPU core
#SBATCH -t 3-0:00:00
#SBATCH --mem=40G


# conda init bash
source /home/x_scorb/miniconda3/etc/profile.d/conda.sh
conda activate htr

# Parameters
main_script=/proj/document_analysis/users/x_scorb/codes/htr-seq2seq-bressay/src/eval.py

echo "Create dir for log"
CURRENTDATE=`date +"%Y-%m-%d"`
echo "currentDate :"
echo $CURRENTDATE


output_file="${PATHLOG}/${SLURM_JOB_ID}.txt"

export PYTHONPATH=/proj/document_analysis/users/x_scorb/codes/htr-seq2seq-bressay/

# The job
# -u : Force les flux de sortie et d'erreur standards à ne pas utiliser de tampon. Cette option n'a pas d'effet sur le flux d'entrée standard
python -u $main_script \
  --dir_data "/proj/document_analysis/users/x_scorb/data/bressay_final/data_split/lines" \
  --num_workers 4 \
>> $output_file


