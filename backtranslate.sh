#!/bin/bash
#SBATCH --job-name=ml_scratch
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --time 1-08:00:00
#SBATCH --signal=B:HUP@600

# Setup Environment and variables
source ../venv/bin/activate

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="/home2/$USER"

mkdir -p $LOCAL_ROOT/{data,ml_checkpoints,ml_results}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/ml_checkpoints
RESULTS=$LOCAL_ROOT/ml_results

export ILMULTI_CORPUS_ROOT=$DATA


# Set other variables.
DATASET='complete-en-ml'
EXPORT_DIR='trained_ml_ckpts'

ARCH='transformer'
EPOCH=64


# Copy Checkpoints and Datset
rsync -rvz $REMOTE_ROOT/datasets/$DATASET/ $DATA/$DATASET/


# Setup for exporting
function _export {
    ssh $USER@ada "mkdir -p ~/$EXPORT_DIR/"
    ssh $USER@ada "mkdir -p ~/ml_results/"
    rsync -rvz $RESULTS/ ~/ml_results/
    rsync -rvx $CHECKPOINTS/checkpoint_{last,best}.pt ~/$EXPORT_DIR/
}
trap "_export" SIGHUP


# Pre-process
python3 preprocess_cvit.py config.yaml

# Functions
function train_mt {
    python3 train.py \
        --task shared-multilingual-translation \
        --num-workers 0 \
        --arch $ARCH \
        --max-tokens 5000 --lr 1e-4 --min-lr 1e-9 \
        --max-epoch $EPOCH \
        --optimizer adam \
        --save-dir $CHECKPOINTS \
        --log-format simple --log-interval 200 \
        --criterion label_smoothed_cross_entropy \
        --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 \
        --ddp-backend no_c10d \
        --update-freq 2 \
        --reset-optimizer \
        --share-all-embeddings \
        config.yaml 
}

function _backtranslate {
    python3 generate.py config.yaml \
        --task shared-multilingual-translation  \
        --path $CHECKPOINTS/checkpoint_best.pt > $RESULTS/final_translation.txt
    cat "$RESULTS/final_translation.txt" \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > "$RESULTS/complete.hyp"
    cat "$RESULTS/final_translation.txt" \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > "$RESULTS/complete.ref"
    split -d -l 2886 "$RESULTS/complete.hyp" "$RESULTS/hyp."
    split -d -l 2886 "$RESULTS/complete.ref" "$RESULTS/ref."
}

# Run the function
train_mt
_backtranslate
# Export Results
wait
_export
