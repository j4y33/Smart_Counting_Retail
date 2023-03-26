import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from smart_counter.utils import cosine_distance_vectorized
from model_training.get_metrics import get_embeddings_all
from argparse import ArgumentParser
from pathlib import Path
import cv2
import numpy as np
from argparse import ArgumentParser

from torchvision.transforms import Normalize, ToTensor, Resize, Compose
import torch
from torchreid import models
from torchreid.utils import FeatureExtractor
from PIL import Image


def get_embeddings_all_from_folder(folder_path: Path):
    dfs = []
    for folder in folder_path.glob('*'):
        df = pd.read_json(folder/'embeddings.json', lines=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    nump = df.embedding.to_numpy()
    nump = np.array([np.array(x) for x in nump])
    return nump


def get_cos_dist(crops_folder: str,
                 model_name: str,
                 device: str,
                 weights: str,
                 width: int = 128,
                 height: int = 256):
    """
    get optimal cosine threshold distance
    """

    crops_folder = Path(crops_folder)
    embeddings_all = get_embeddings_all_from_folder(crops_folder)

    num = len(list(crops_folder.glob('*')))
    clusters_num = 0
    distance_threshold = 0.2

    print(num)
    print()
    while abs(num-clusters_num) != 0:
        clustering = AgglomerativeClustering(n_clusters=None,
                                             affinity='cosine',
                                             distance_threshold=distance_threshold,
                                             linkage='average').fit_predict(embeddings_all)
        clusters_num = len(np.unique(clustering))
        print(f'clusters {clusters_num} | cur cos dist:  {distance_threshold}')
        distance_threshold += 0.01

    print(distance_threshold)
    print(min_samples)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--crops_folder', type=str, required=False,
                        default='model_training/crops/crops_all',  help='Path to query folder')
    parser.add_argument('--model_name', type=str, required=False,
                        default='osnet_ain_x1_0',
                        help='name of model')
    parser.add_argument('--device', type=str, required=False,
                        default='cuda:0',
                        help='device to use')
    parser.add_argument('--weights', type=str, required=False,
                        default='model_training/models/resnet50Triplet_4_328/model/model.pth.tar-10',
                        help='Path to model weights')
    parser.add_argument('--width', type=int, required=False,
                        default=128,
                        help='img width')
    parser.add_argument('--heigth', type=int, required=False,
                        default=256,
                        help='img heigth')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    get_cos_dist(
        args.crops_folder,
        args.model_name,
        args.device,
        args.weights,
        args.width,
        args.heigth
    )
