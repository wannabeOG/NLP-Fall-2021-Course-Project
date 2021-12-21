#!/bin/bash
#SBATCH --job-name=bt-mr
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --time 0-08:00:00
#SBATCH --signal=B:HUP@600

# assuming $DATA/monolingual/mr/monolingual.mr
# assuming $DATA/$DATASET/ (the full parallel dataset)
# Make sure that $DATASET is registered in corpora.py

# Constants
DATASET="pib-en-mar"

# Basic Setup
source ../venv/bin/activate
LOCAL_ROOT="/ssd_scratch/cvit/$USER"
REMOTE_ROOT="/home2/$USER"
mkdir -p $LOCAL_ROOT/{data,checkpoints}
DATA=$LOCAL_ROOT/data
CHECKPOINTS=$LOCAL_ROOT/checkpoints
CKPT=$CHECKPOINTS/checkpoint_last.pt
export ILMULTI_CORPUS_ROOT=$DATA
mkdir -p $DATA/monolingual/mr

function _copy {
    rsync -rvz $REMOTE_ROOT/checkpoints/checkpoint_{last,best}.pt $CHECKPOINTS/
    rsync -rvz $REMOTE_ROOT/datasets/$DATASET/ $DATA/$DATASET/
    rsync -rvz $REMOTE_ROOT/datasets/monolingual/mr/ $DATA/monolingual/mr/
}

_copy

mv $DATA/$DATASET/test.mr-en.mr "$DATA/$DATASET/test.mr-en.mr_original"
mv $DATA/$DATASET/test.mr-en.en "$DATA/$DATASET/test.mr-en.en_original"

function _backtranslate {
    python3 generate.py config.yaml \
        --task shared-multilingual-translation  \
        --skip-invalid-size-inputs-valid-test \
        --path $CKPT > $DATA/monolingual/mr/_.txt
    cat $DATA/monolingual/mr/_.txt \
        | grep "^H" | sed 's/^H-//g' | sort -n | cut -f 3 \
        | sed 's/ //g' | sed 's/_/ /g' | sed 's/^ //g' \
            > $DATA/monolingual/mr/bt.hyp
    cat $DATA/monolingual/mr/_.txt \
        | grep "^T" | sed 's/^T-//g' | sort -n | cut -f 2 \
        | sed 's/ //g' | sed 's/_/ /g' | sed 's/^ //g' \
            > $DATA/monolingual/mr/bt.ref
}

# cp $DATA/monolingual/mr/monolingual.mr "$DATA/$DATASET/test.mr-en.mr"
# touch "$DATA/$DATASET/test.mr-en.en"
# _backtranslate
# mv $DATA/monolingual/mr/bt.hyp $DATA/monolingual/mr/bt.en
# mv $DATA/monolingual/mr/bt.ref $DATA/monolingual/mr/clean_monolingual.mr

cp $DATA/monolingual/mr/bt.en "$DATA/$DATASET/test.mr-en.en"
touch "$DATA/$DATASET/test.mr-en.mr"
_backtranslate
mv $DATA/monolingual/mr/bt.hyp $DATA/monolingual/mr/bt_bt.mr

# Cleanup backtranslate
rm -rf $DATA/monolingual/mr/bt.ref
rm -rf $DATA/monolingual/mr/_.txt

# Get scores
fairseq-score -s $DATA/monolingual/mr/bt_bt.mr -r $DATA/monolingual/mr/clean_monolingual.mr --sentence-bleu > $DATA/monolingual/mr/scores.txt

# Export
ssh $USER@ada "mkdir -p $REMOTE_ROOT/datasets/monolingual"
rsync -rvz $DATA/monolingual/mr/{clean_monolingual.mr,bt.en,scores.txt} $REMOTE_ROOT/datasets/monolingual/mr/
rm -rf $LOCAL_ROOT
