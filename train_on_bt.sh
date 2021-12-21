#!/bin/bash
#SBATCH --job-name=rad_ml
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=30G
#SBATCH --time 1-01:00:00
#SBATCH --signal=B:HUP@600

# Setup Environment and variables
source ../venv/bin/activate

LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="/home2/$USER"

# cleanup previous iters
rm -rf $LOCAL_ROOT

mkdir -p $LOCAL_ROOT/{data,checkpoints,results}

DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
RESULTS=$LOCAL_ROOT/results

export ILMULTI_CORPUS_ROOT=$DATA


# Set other variables.
DATASET='complete-en-ml'
EXPORT_DIR='scratch_ml_ckpts'

ARCH='transformer'
EPOCH=50

# Copy Checkpoints and Datset
rsync -rvz $REMOTE_ROOT/scratch_ml_ckpts/checkpoint_{best,last}.pt $CHECKPOINTS/
# mv $CHECKPOINTS/checkpoint_best.pt $CHECKPOINTS/checkpoint_last.pt
rsync -rvz $REMOTE_ROOT/datasets/$DATASET/ $DATA/$DATASET/


# Setup for exporting
function _export {
    ssh $USER@ada "mkdir -p ~/$EXPORT_DIR/"
    ssh $USER@ada "mkdir -p ~/scratch_ml_results/"
    rsync -rvz $RESULTS/ ~/scratch_ml_results/
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

function _evaluate {
    # Evaluate ckpt_best
    printf "\n\nEVALUATING ON CKPT_BEST\n------------------------------------\n"
    python3 generate.py config.yaml \
        --task shared-multilingual-translation  \
        --path $CHECKPOINTS/checkpoint_best.pt > $RESULTS/checkpoint_best_translations.txt
    cat "$RESULTS/checkpoint_best_translations.txt" \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > "$RESULTS/complete.best.hyp"
    cat "$RESULTS/checkpoint_best_translations.txt" \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > "$RESULTS/complete.best.ref"
    split -d -l 2886 "$RESULTS/complete.best.hyp" "$RESULTS/best.hyp."
    split -d -l 2886 "$RESULTS/complete.best.ref" "$RESULTS/best.ref."
    python3 -m wateval.evaluate \
        --hypothesis "$RESULTS/best.hyp.00" \
        --references "$DATA/$DATASET/test.ml-en.ml"

    # python3 -m wateval.evaluate \
    #    --hypothesis "$RESULTS/best.hyp.01" \
    #    --references "$DATA/$DATASET/test.mr-en.en"

    # Evaluate ckpt_last
    printf "\n\nEVALUATING ON CKPT_LAST\n-----------------------\n"
    python3 generate.py config.yaml \
        --task shared-multilingual-translation  \
        --path $CHECKPOINTS/checkpoint_last.pt > $RESULTS/checkpoint_last_translations.txt
    cat "$RESULTS/checkpoint_last_translations.txt" \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > "$RESULTS/complete.last.hyp"
    cat "$RESULTS/checkpoint_last_translations.txt" \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/▁/ /g' | sed 's/^ //g' \
            > "$RESULTS/complete.last.ref"
    split -d -l 2886 "$RESULTS/complete.last.hyp" "$RESULTS/last.hyp."
    split -d -l 2886 "$RESULTS/complete.last.ref" "$RESULTS/last.ref."
    python3 -m wateval.evaluate \
        --hypothesis "$RESULTS/last.hyp.00" \
        --references "$DATA/$DATASET/test.ml-en.ml"

    # python3 -m wateval.evaluate \
        # --hypothesis "$RESULTS/last.hyp.01" \
        # --references "$DATA/$DATASET/test.mr-en.en"
}

# train_mt
_evaluate
_export
