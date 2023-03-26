from pathlib import Path
import pandas as pd
from smart_counter.utils import cosine_distance_vectorized
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import classification_report
from argparse import ArgumentParser


def report_1_or_2(folder: str,
                  cos_range_threshold: float,
                  k2_threshold: float):
    """
    get metrics of predicting 1 or 2 people in cluster
    cluster with 2 people naming: cluster1id_cluster2id
    """
    folder = Path(folder)
    y_true = []
    y_pred = []
    for track in folder.glob('*'):
        if len(list(track.glob('*.jpg'))) < 8:
            continue
        df = pd.read_json(track/'embeddings.json', lines=True)
        nump = df.embedding.to_numpy()
        nump = np.array([np.array(x) for x in nump])

        distance_matrix = cosine_distance_vectorized(nump, nump)
        k2, p = stats.normaltest(distance_matrix, axis=None)

        range_m = np.min(distance_matrix)
        range_max = np.max(distance_matrix)
        cos_range = range_max-range_m
        if '_' in track.name:
            y_true.append(2)
        else:
            y_true.append(1)
        if cos_range > cos_range_threshold or k2 > k2_threshold:
            y_pred.append(2)
        else:
            y_pred.append(1)

    print(classification_report(y_true, y_pred, labels=[1, 2], target_names=['one', 'two']))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--folder', type=str, required=False,
                        default='model_training/crops/crops_random_merged',  help='Path to folder')
    parser.add_argument('--cos_range_threshold', type=float, required=False,
                        default=0.55,
                        help='Path to folder with crops')
    parser.add_argument('--k2_threshold', type=float, required=False, default=1000,
                        help='percent for test tracks')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    report_1_or_2(args.folder, args.cos_range_threshold, args.k2_threshold)
