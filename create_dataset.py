import numpy as np
import pandas as pd
from tqdm import tqdm, trange 
import json
import argparse
import random
import csv
import random

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
        default="random",
        type=str,
        help="The number of acquired data size",
    )
    args = parser.parse_args()
    return args

# Function to read IDs from CSV and store them in a list
def read_ids_from_csv(csv_file):
    ids_list = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            # Assuming IDs are in the first column (index 0)
            ids_list.append(int(row[0]))
    return ids_list


if __name__ == '__main__':
    '''
    Suppose all the data is in the folder ./X, where X = {AGNews, IMDB, TREC, Yahoo, Yelp-full}
    '''
    args = get_arguments()

    dataset = args.dataset
    n_sample = int(args.n_sample)
    method = args.method

    range_ = {"imdb": 25000, "trec": 5452, "agnews": 120000, "yelp-full": 650000}
    print(range_[dataset])
    print(f"Creating dataset for : {dataset} with {n_sample} samples")

    main_file = f"./{dataset}/unlabeled.json"
    train_json_file = f'./{dataset}/{method}_1/train_{n_sample}.json'
    val_json_file = f'./{dataset}/{method}_1/valid_{n_sample}.json'

    if method == "patron":
        sample_file = f"./{dataset}/train_idx_roberta-base_round1_rho0.01_gamma0.5_beta0.5_mu0.5_{n_sample}.json"
        with open(sample_file, 'r') as sample_ids_file:
            sample_ids = json.load(sample_ids_file)
    elif method == "random":
        sample_ids = random.sample(range(range_[dataset]), n_sample)
    else:
        sample_file = f"./{dataset}/sample_dataset/{method}/{method}{n_sample}.csv"
        sample_ids = read_ids_from_csv(sample_file)

    # Read data from the unlabelled JSON file
    with open(main_file, 'r') as data_file:
        data = [json.loads(line) for line in data_file]
    
    print(type(data[0]))
    # # Filter data based on sample IDs
    # # filtered_data = [datapoint for datapoint in data if str(datapoint['id']) in sample_ids]
    # filtered_data = [datapoint for idx, datapoint in enumerate(data) if idx in sample_ids]
    # sample_ids = [623, 762, 692, 3422, 3970, 2124, 3853, 4104, 464, 2325, 3938, 1352, 1371, 893, 4436, 553, 3662, 1473, 3593, 1933, 2915, 998, 3117, 993, 5256, 3062, 4701, 3373, 4820, 2330, 3850, 2673, 191, 265, 5304, 2052, 788, 1742, 5224, 4936, 3081, 3238, 1147, 2996, 3937, 798, 83, 390, 1150, 2196, 3396, 2547, 5399, 4894, 3666, 2528, 319, 64, 2303, 1978, 691, 3319, 4405, 1628, 1605, 683, 1921, 1804, 2620, 867, 1075, 1513, 2745, 5193, 3697, 4637, 339, 5291, 1768, 4712, 3413, 3967, 1642, 2061, 3540, 1817, 1499, 4690, 3446, 2752, 1874, 1934, 676, 1595, 3914, 3667, 2085, 1877, 2165, 2973, 2410, 1062, 2505, 3791, 1442, 4070, 551, 3985, 822, 5284, 1194, 4438, 133, 3011, 981, 3219, 5351, 3980, 2338, 4378, 2055, 4461, 2681, 1030, 2385, 4750, 3288, 2567, 1500, 4556, 1983, 4599, 4428, 2828, 741, 3601, 1751, 5416, 2971, 3602, 1924, 4546, 5360, 484, 2097, 1455, 4755, 3195, 2878, 3863, 3567, 4452, 669, 5091, 5437, 3441, 3682, 4706, 5151, 1987, 5357, 86, 3229, 204, 2142, 108, 1028, 2096, 955, 1160, 4253, 411, 1757, 2104, 2123, 1930, 5440, 454, 2226, 3626, 1861, 2910, 2926, 1143, 1116, 4498, 638, 1588, 3635, 2924, 3100, 1793, 1556, 2870, 2033, 3414, 4402, 2290, 4063, 580, 1849, 5047, 1675, 943, 2357, 524, 4868, 2998, 3671, 5145, 4531, 4410, 2585, 4823, 4052, 2514, 5045, 4345, 1799, 4206, 293, 5294, 438, 4092, 1123, 2190, 3704, 1567, 5364, 1579, 1216, 3139, 3652, 3154, 3538, 4244, 1575, 4543, 4337, 146, 4789, 1333, 1568, 713, 1671, 2261, 2731, 709, 3147, 448, 2510, 1219, 4719, 3484, 2827, 2643, 1664, 1585, 1478, 253, 1889, 5124, 4314, 1772, 3366, 4075, 2464, 2569, 137, 1267, 2795, 3562, 455, 860, 4385, 249, 269, 2126, 873, 4156, 5177, 3696, 2512, 5158, 5105, 2283, 1347, 3843, 3341, 2007, 3364, 2899, 3421, 5029, 1560, 5445, 605, 4809, 3913, 975, 4776, 4473, 2130, 675, 786, 1502, 2418, 1784, 5121, 4487, 1622, 188, 478, 3691, 4583, 5251, 3259, 1519, 4447, 5210, 56, 1835, 4641, 2220, 118, 3619, 4950, 4906, 1366, 1155, 183, 2556, 2487, 2383, 685, 1794, 3574, 2847, 2483, 311, 3710, 2024, 672, 3180, 1712, 1674, 2946, 3209, 2596, 5405, 4601, 1783, 1974, 2704, 782, 5048, 3424, 5423, 1003, 3624, 1283, 650, 815, 3878, 4274, 4763, 936, 1620, 420, 1445, 4593, 3199, 582, 5432, 168, 3368, 3051, 55, 2496, 1661, 3579, 3401, 805, 2439, 1332, 1570, 4536, 337, 1429, 2537, 217, 876, 1117, 4647, 2167, 972, 4769, 3303, 4766, 4280, 4917, 1707, 1859, 2169, 1701, 5058, 3145, 3733, 306, 3400, 4079, 1324, 1264, 966, 894, 1756, 1493, 3659, 2367, 5189, 3064, 2504, 2589, 3630, 1779, 3466, 542, 112, 4276, 3178, 340, 3713, 3116, 4089, 1990, 4138, 3392, 1465, 2696, 4421, 4309, 2211, 1619, 1162, 712, 1621, 3515, 3798, 2845, 501, 4916, 1714, 4300, 4848, 5178, 150, 1449, 16, 689, 3590, 4372, 3183, 4204, 1006, 1359, 2113, 4900, 3715, 2949, 2180, 2862, 1561, 2161, 278, 1549, 824, 467, 978, 2107, 2044, 4852, 1643, 2011, 2856, 3497, 5190, 1291, 4169, 1697, 988, 4604, 5315, 2241, 3448, 2763, 2629, 4544, 1717, 5150, 3746, 2041, 3022, 4228, 4407, 1177, 1896, 4358, 482, 3694, 266, 4708, 3765]
    filtered_data = []
    for idx in sample_ids:
            filtered_data.append(data[idx])


    # Write JSON to file
    with open(train_json_file, 'w', encoding='utf-8') as jsonfile:
        for item in filtered_data:
            json.dump(item, jsonfile, ensure_ascii=False)
            jsonfile.write('\n')

    # Sample 128 points from main_file excluding IDs from csv_file and store them in val.json
    val_data = []
    unsampled_data = []
    for idx, datapoint in enumerate(data):
        if idx + 1 in sample_ids:
            continue
        unsampled_data.append(datapoint)

    print(n_sample)
    random_samples = random.sample(unsampled_data, min(n_sample, len(unsampled_data)))  # Ensure not sampling more than available data
    for row in random_samples:
        val_data.append(row)

    # Write sampled data to val.json with each line as a JSON object
    with open(val_json_file, 'w', encoding='utf-8') as val_jsonfile:
        for item in val_data:
            json.dump(item, val_jsonfile, ensure_ascii=False)
            val_jsonfile.write('\n')

    print(f"Created: {train_json_file} and {val_json_file}")
