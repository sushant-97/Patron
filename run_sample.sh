task=$1

n_samples_=$2
batch_size=32
k_=50
rho_=0.01
gamma_=0.5
beta_=0.5

generate_embeddings_cmd="python gen_embedding_simcse.py \
            --dataset ${task} \
            --gpuid 0 "

data_selection_cmd="python patron_sample.py \
            --dataset ${task} \
            --k ${k_} --rho ${rho_} --gamma ${gamma_} --beta ${beta_} --n_sample ${n_samples_}"

# echo $generate_embeddings_cmd
# eval $generate_embeddings_cmd

echo $data_selection_cmd
eval $data_selection_cmd