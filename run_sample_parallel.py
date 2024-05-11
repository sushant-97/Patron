import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

if_methods = ["patron"]
train_labels = [128, 256, 512]

if len(sys.argv) != 2:
    print("Usage: python script_name.py <dataset_name>")
    sys.exit(1)

dataset_name = sys.argv[1]

def run_command(train_label, if_method):
    command = f"bash run_sample.sh {dataset_name} {train_label} > logs/{dataset_name}_{if_method}_{train_label}.txt"
    print(command)
    process = subprocess.Popen(command, shell=True)
    process.wait()

# Using ThreadPoolExecutor for parallel execution
with ThreadPoolExecutor() as executor:
    for train_label in train_labels:
        for if_method in if_methods:
            executor.submit(run_command, train_label, if_method)
