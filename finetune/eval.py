import numpy as np
import argparse
from sklearn.metrics import classification_report
from collections import Counter

''' loading training arguments '''
def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--dataset",
        default='trec',
        type=str,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--n_sample",
        default=512,
        type=int,
        help="The number of acquired data size",
    )

    parser.add_argument(
        "--method",
        default="r",
        type=str,
        help="The number of acquired data size",
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    '''
    Suppose all the data is in the folder ./X, where X = {AGNews, IMDB, TREC, Yahoo, Yelp-full}
    '''
    args = get_arguments()

    dataset = args.dataset
    train_labels = int(args.n_sample)
    method = args.method
    if method == "r" or method == "b" or method == "c":
        pass

    label_file = f"./{dataset}/{method}/{train_labels}/cache/labels.txt"
    prediction_file = f"./{dataset}/{method}/{train_labels}/cache/prediction.txt"

    # Read the data from the files
    with open(label_file, 'r') as f:
        labels = np.array([line.strip() for line in f.readlines()])

    with open(prediction_file, 'r') as f:
        predictions = np.array([line.strip() for line in f.readlines()])
    
    unique_labels = list(np.unique(labels))

    count_labels = Counter(labels)
    count_preds = Counter(predictions)
    report = classification_report(labels, predictions, target_names=unique_labels)

    # Output the results to a text file
    with open(f'performance_metrics_{dataset}.txt', 'a') as f:
            # Print the counts for each unique element
        # for item, frequency in count_preds.items():
        #     f.write(f"Class: {item} || Acutal: {count_labels[item]} || Predictions: {frequency}\n")

        if method == "r" or method == "b" or method == "c":
            f.write(f"Classification report for {dataset} IsoForest {method} {train_labels}:\n")
        else:
            f.write(f"Classification report for {dataset} {method} {train_labels}:\n")
        f.write(f"Overall Accuracy: {report}\n")

    print("Performance metrics have been calculated and stored in performance_metrics.txt.")
