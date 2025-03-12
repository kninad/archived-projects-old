#!/bin/sh

COUNTER=0
trn_folder="book_small"
trn_file="book_small/book_train_hb.csv"
#trn_folder="exp2.2_int_taxo"    # for use in exp_name
#trn_file="exp2.2_int_taxo/book_train_hb.csv"    # for dset arg param 

#for dim in 30; do # emd dim
#for lr in 0.1; do # learning_rate
#done
#done

dim=10
lr=0.1

epochs=10000
negs=20
burnin=5
batchsize=512
eval_each=1
train_threads=4
gpu=-1

echo "Submitting Job"
echo "************************"
COUNTER=$((COUNTER + 1))

exp_name="$(date "+%Y-%m-%d_%H-%M-%S")_${trn_folder}_dim${dim}_lr${lr}_epoc${epochs}_neg${negs}_burn${burnin}"

echo "Exp name: ${exp_name}"

logpath="./log/${exp_name}/"
chkpath="./log/${exp_name}/book_model.pth"
mkdir ${logpath}

# Call the python script
#source activate poincare
python3 embed.py \
       -dim ${dim} \
       -lr ${lr} \
       -epochs ${epochs} \
       -negs ${negs} \
       -burnin ${burnin} \
       -train_threads ${train_threads} \
       -gpu ${gpu} \
       -dset "./data/book_data/${trn_file}" \
       -logfolder ${logpath} \
       -checkpoint ${chkpath} \
       -fresh \
       -sparse \
       -manifold poincare




