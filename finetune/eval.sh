dataset=trec
# Define method and n_sample values
methods=("r" "b" "c")
n_samples=(128 256 512)

# Loop through method and n_sample combinations
for method in "${methods[@]}"; do
    for sample_size in "${n_samples[@]}"; do
        echo "Running eval.py with method=$method and n_sample=$sample_size"
        python eval.py --dataset $dataset --n_sample $sample_size --method $method
        echo "Completed eval.py with method=$method and n_sample=$sample_size"
    done
done