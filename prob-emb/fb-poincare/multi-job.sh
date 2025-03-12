#!/bin/sh

COUNTER=0

trn_folder="exp2.2_int_taxo"    # for use in exp_name
trn_file="exp2.2_int_taxo/book_train_hb.csv"    # for dset arg param 

for dim in 10 15; do # emd dim
for lr in 0.1 0.2 0.3; do # learning_rate

epochs=5000
negs=50
burnin=10
batchsize=128
eval_each=1
train_threads=24
gpu=-1   # -1 means no gpu


echo "Submitting Job"
echo "************************"
COUNTER=$((COUNTER + 1))

exp_name="$(date "+%Y-%m-%d_%H-%M-%S")_${trn_folder}_dim${dim}_lr${lr}_epoc${epochs}_neg${negs}_burn${burnin}"

echo "Exp name: ${exp_name}"

logpath="./log/${exp_name}/"
chkpath="./log/${exp_name}/book_model.pth"
mkdir ${logpath}

runFile="run_${trn_folder}_${COUNTER}.sh"
touch ${runFile}

echo "#!/bin/bash" > ${runFile}
#echo "source activate tfcpu-py27" >> ${runFile}
echo "python3 embed.py \\" >> ${runFile}
echo "-dim ${dim} -lr ${lr} -epochs ${epochs} -negs ${negs} \\" >> ${runFile}
echo "-burnin ${burnin} -train_threads ${train_threads} -gpu ${gpu} \\" >> ${runFile}
echo "-dset "./data/book_data/${trn_file}" \\" >> ${runFile}
echo "-logfolder ${logpath} \\" >> ${runFile}
echo "-checkpoint ${chkpath} \\" >> ${runFile}
echo "-fresh -sparse -manifold poincare" >> ${runFile}

#cat ${runFile}
sbatch --partition titanx-long --mem 30000 --gres=gpu:1 ${runFile}
echo "Done calling the script"
echo "************************"

done
done


