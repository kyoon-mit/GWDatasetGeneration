#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --time=12:00:00
#SBATCH --job-name=gen_bns
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=40G
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/GWDatasetGeneration/slurm_logs/output-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/GWDatasetGeneration/slurm_logs/error-%j.err

TRAIN_CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/GWDatasetGeneration/configs/ai4gw@cern/bns_snr_5_50_powerlaw_256Hz/BNS_SNR5-50_TRAIN_VAL_256Hz_BNS_powerlaw_spin.yaml
VAL_CONFIG=$TRAIN_CONFIG
TEST_CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/GWDatasetGeneration/configs/ai4gw@cern/bns_snr_5_50_powerlaw_256Hz/BNS_SNR5-50_TEST_256Hz_BNS_powerlaw_spin.yaml

TRAIN_BKG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3b_H1_L1_4096Hz
VAL_BKG=$TRAIN_BKG
TEST_BKG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz

DATA_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/ai4gw@cern/bns_snr_5_50_powerlaw_256Hz_80K_10K_100K
TRAIN_DATA=$DATA_DIR/train
VAL_DATA=$DATA_DIR/val
TEST_DATA=$DATA_DIR/test

source /n/home04/kyoon/miniforge3/etc/profile.d/conda.sh
conda activate bns

echo "Running on host: $(hostname)"

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/GWDatasetGeneration

python main.py --config $VAL_CONFIG   --data $VAL_BKG   --out $VAL_DATA   --num-waveforms 10000
python main.py --config $TEST_CONFIG  --data $TEST_BKG  --out $TEST_DATA  --num-waveforms 100000
python main.py --config $TRAIN_CONFIG --data $TRAIN_BKG --out $TRAIN_DATA --num-waveforms 80000
python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/GWDatasetGeneration/combine_h5.py $DATA_DIR
cp $TRAIN_CONFIG $DATA_DIR
cp $VAL_CONFIG $DATA_DIR
cp $TEST_CONFIG $DATA_DIR