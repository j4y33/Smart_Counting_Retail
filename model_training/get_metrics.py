import random
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from torchvision.transforms import Normalize, ToTensor, Resize, Compose
import torch
from torchreid import models
from torchreid.utils import FeatureExtractor
from torchreid.metrics.rank import eval_market1501
from PIL import Image
from typing import Union


from smart_counter.utils import cosine_distance_vectorized


def get_imgs_person_num(folder: str):
    """
    count pid num in folder

    Parameters
    ----------
    folder
        path to folder
    """
    folder = Path(folder)
    person_num = 0
    imgs_num = 0
    for f in folder.glob('*'):
        for img in f.glob('*'):
            pid = int(img.name.split('_')[0])
            person_num = pid if pid > person_num else person_num
            imgs_num += 1
    return imgs_num, person_num


def get_img(img_path: Union[str, Path]):
    """
    return cv2 rgb img
    """
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_path.name, img


def get_embedding(img, model, device, transform):
    with torch.no_grad():
        x = transform(Image.fromarray(img))
        x = x.unsqueeze(0).to(device)
        embeddings = model(x).cpu().numpy()
        return embeddings[0]


def get_embeddings_all(imgs_pathes, model, device, transform):
    res = []
    for img in imgs_pathes:
        img = cv2.imread(str(img))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res.append(get_embedding(img, model, device, transform))
    return np.array(res)


def get_model_metrics(query_folder: str,
                      gallery_folder: str,
                      model_name: str,
                      device: str,
                      weights: str,
                      width: int = 128,
                      height: int = 256):
    """
    get mAP, accuracy, average cosine distance, max cosine distance, min cosine distance of ready model
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]

    query_folder = Path(query_folder)
    gallery_folder = Path(gallery_folder)
    input_shape = (height, width)

    model = FeatureExtractor(
        model_name=model_name,
        model_path=weights,
        device=device
    )
    transform = Compose([
        Resize(input_shape),
        ToTensor(),
        Normalize(mean=_MEAN, std=_STD),
    ])

    imgs_pathes_gallery = list(gallery_folder.glob('*'))
    imgs_pathes_query = list(query_folder.glob('*'))

    embeddings_all_gallery = get_embeddings_all(imgs_pathes_gallery, model, device, transform)

    av = []
    for i in tqdm(imgs_pathes_query):
        img_name, img = get_img(i)
        embedding = get_embedding(img, model, device, transform)
        embedding = np.expand_dims(embedding, 0)

        cosine = cosine_distance_vectorized(embedding, embeddings_all_gallery)
        max_index_col = np.argmin(cosine, axis=1)

        res_name = imgs_pathes_gallery[max_index_col[0]].name
        if img_name.split('_')[0] == res_name.split('_')[0]:
            av.append(cosine[0][max_index_col])

    embeddings_all_query = get_embeddings_all(imgs_pathes_query, model, device, transform)

    distance_matrix = cosine_distance_vectorized(embeddings_all_query, embeddings_all_gallery)
    q_pids = np.array([int(i.name.split('_')[0]) for i in imgs_pathes_query])
    g_pids = np.array([int(i.name.split('_')[0]) for i in imgs_pathes_gallery])
    q_camids = np.full((len(imgs_pathes_query)), 1, dtype=int)
    g_camids = np.full((len(imgs_pathes_gallery)), 2, dtype=int)
    max_rank = 1
    _, mAP = eval_market1501(distance_matrix, q_pids, g_pids, q_camids, g_camids, max_rank)

    print(f"mAP: {mAP}")
    print(f"accuracy: {len(av)/len(imgs_pathes_query)}")
    print(f"average cosine distance: {np.average(av)}")
    print(f"max cosine distance: {np.max(av)}")
    print(f"min cosine distance: {np.min(av)}")

    return (len(av)/len(imgs_pathes_query), np.average(av), np.max(av), np.min(av), mAP)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--query_folder', type=str, required=False,
                        default='model_training/crops/dataset_4_328/query',  help='Path to query folder')
    parser.add_argument('--gallery_folder', type=str, required=False,
                        default='model_training/crops/dataset_4_328/gallery',
                        help='Path to gallery folder')
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

    get_model_metrics(
        args.query_folder,
        args.gallery_folder,
        args.model_name,
        args.device,
        args.weights,
        args.width,
        args.height
    )
