from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import itertools
from argparse import ArgumentParser
from model_training.get_metrics import get_img


def get_av_size(path: str):
    """
    get average image size in folder
    """
    path = Path(path)
    imgs_pathes = list(itertools.chain.from_iterable([list(f.glob('*')) for f in path.glob('*')]))
    shapes = [[w, h] for h, w, _ in map(lambda x: get_img(x)[1].shape, imgs_pathes)]
    av = np.average(shapes, axis=0)
    print("average crop width: ", av[0])
    print("average crop heigth: ", av[1])


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=False,
                        default='model_training/crops/dataset_4_328',  help='Path to dataset folder')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    get_av_size(args.root)
