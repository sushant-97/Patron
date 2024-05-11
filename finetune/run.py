import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

trial = 'R3'
# Define the if_method and train_label values
if_methods = ['b']
# if_methods = ["random_1","random_2", "random_3"]
train_labels = [32,64,128, 256, 512]

# methods=("r" "b" "c")
# n_samples=(128 256 512)

if len(sys.argv) != 2:
    print("Usage: python script_name.py <dataset_name>")
    sys.exit(1)

dataset = sys.argv[1]
print(dataset)

def run_command(train_label, if_method):
    command = f"bash commands/run_updated.sh {dataset} {train_label} {if_method} {trial} > logs/{dataset}/{dataset}_{trial}_{if_method}_{train_label}.txt"
    print(command)
    process = subprocess.Popen(command, shell=True)
    process.wait()

# for train_label in train_labels:
#     for if_method in if_methods:
#         # Construct the command to be executed
#         command = f"bash commands/run_updated.sh {dataset} {train_label} {if_method} > logs_/{dataset}/{dataset}_{if_method}_{train_label}_4.txt"
#         print(command)
        
#         # Execute the command using subprocess
#         process = subprocess.Popen(command, shell=True)
        
#         # Wait for the process to finish before starting the next one
#         process.wait()

# Using ThreadPoolExecutor for parallel execution
with ThreadPoolExecutor() as executor:
    for train_label in train_labels:
        for if_method in if_methods:
            executor.submit(run_command, train_label, if_method)