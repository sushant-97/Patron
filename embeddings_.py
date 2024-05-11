import pickle
import numpy as np

def load_data(dataset='trec', embedding_model='roberta-base', template_id=0):
    path = f'{dataset}/'
    with open(path + f'embedding_simcse_unsup-simcse-roberta-base_unlabeled.pkl', 'rb') as f:
        train_emb = pickle.load(f)
    return train_emb

def save_embeddings_to_file(embeddings, dataset="trec", filename='traincorpus_embedding.txt'):
    filename = f"./{dataset}/{filename}"
    # Convert embeddings to numpy array for easier handling
    embeddings = np.array(embeddings)
    # Open file for writing
    with open(filename, 'w') as f:
        # Write the number of datapoints and dimension as the first line
        f.write(f'{len(embeddings)} {len(embeddings[0])}\n')
        # Write embeddings, one per line
        for emb in embeddings:
            # Convert each embedding to a string and join with space
            f.write(' '.join(map(str, emb)) + '\n')

def slice_list(input_list, num_slices=6):
    # Calculate the length of each slice
    slice_length = len(input_list) // num_slices
    superlist = []

    for i in range(num_slices):
        # Calculate start and end indices for slicing
        start = i * slice_length
        if i < num_slices - 1:
            end = start + slice_length
        else:
            # Ensure the last slice contains any remaining elements
            end = len(input_list)
        
        # Append the slice to the superlist
        superlist.append(input_list[start:end])

    return superlist

# Load embeddings
t_embed = load_data(dataset="trec")
slices = {i: [] for i in range(6)}
size = len(t_embed[0]) / 6

for data in t_embed:
    sliced = slice_list(data)
    for i, slice in enumerate(sliced):
        slices[i].append(slice)
    
# print(len(slices[0]))
# print(len(slices[0][0]))
# Save embeddings to text file
# save_embeddings_to_file(t_embed)
for key, value in slices.items():
    save_embeddings_to_file(embeddings= value, dataset="trec", filename=f'traincorpus_embedding_{key}.txt')
