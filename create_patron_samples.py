import subprocess
import sys

# Define the if_method and train_label values
# if_methods = ['r', 'b', 'c']
if_methods = ["patron"]
train_labels = [128, 256, 512]

# methods=("r" "b" "c")
# n_samples=(128 256 512)

if len(sys.argv) != 2:
    print("Usage: python script_name.py <dataset_name>")
    sys.exit(1)

dataset = dataset_name = sys.argv[1]

print(dataset)
for train_label in train_labels:
    for if_method in if_methods:
        # Construct the command to be executed
        command = f"bash run_sample.sh {dataset} {train_label} > logs/{dataset}_{if_method}_{train_label}.txt"
        print(command)
        
        # Execute the command using subprocess
        process = subprocess.Popen(command, shell=True)
        
        # Wait for the process to finish before starting the next one
        process.wait()
