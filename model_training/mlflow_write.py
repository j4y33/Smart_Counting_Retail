import mlflow
import torchreid
import time
from pathlib import Path
import yaml
from argparse import ArgumentParser

from model_training.train_model import train_model
from model_training.get_ranks import get_ranks
from model_training.get_metrics import get_model_metrics, get_imgs_person_num


def expirement_with_train(config_path: str):
    """ Make an experiment and save it to mlflow (train model and test it)

    Args:
        config_path: path to .yml config file
    """
    params = yaml.safe_load(open(config_path, 'r'))

    train_model(params['root'],
                params['width'],
                params['height'],
                params['batch_size'],
                params['engine'],
                params['model'],
                params['optimizer'],
                params['alpha'],
                params['stepsize'],
                params['save_dir'],
                params['max_epoch'],
                margin=params['margin'],
                weight_t=params['weight_t'],
                weight_x=params['weight_x'])

    expirement_without_train(config_path)


def expirement_without_train(config_path: str):
    """ Make an experiment and save it to mlflow (only test ready model)

    Args:
        config_path: path to .yml config file
    """
    params = yaml.safe_load(open(config_path, 'r'))

    mlflow.set_tracking_uri(params['remote_server_uri'])
    mlflow.set_experiment(params['experment_name'])

    query_folder = params['root'] + 'query'
    gallery_folder = params['root'] + 'gallery'

    weights = params['save_dir']+'model/model.pth.tar-'+str(params['max_epoch'])

    imgs_num, person_num = get_imgs_person_num(params['root'])

    tensorboard_files = Path(params['save_dir']).glob("events*")

    model_py = get_ranks(weights, params['root'], params['save_dir'], params['optimizer'],
                         params['alpha'], params['width'], params['height'], params['batch_size'])

    top1_accuracy, av_cosine, max_cosine, min_cosine, mAP = get_model_metrics(
        query_folder,
        gallery_folder,
        params['model'],
        params['device'],
        weights,
        params['width'],
        params['height']
    )

    with mlflow.start_run():
        mlflow.pytorch.log_model(model_py, f"extractor_{params['model']}")
        mlflow.log_param("model", params['model'])
        mlflow.log_param("engine", params['engine'])
        mlflow.log_param("max_epoch", params['max_epoch'])
        mlflow.log_param("optimizer", params['optimizer'])
        mlflow.log_param("width", params['width'])
        mlflow.log_param("height", params['height'])
        mlflow.log_param("alpha", params['alpha'])
        mlflow.log_param("stepsize", params['stepsize'])
        mlflow.log_param("imgs_num", imgs_num)
        mlflow.log_param("person_num", person_num)
        if params['engine'] == 'ImageTripletEngine':
            mlflow.log_param("margin", params['margin'])
            mlflow.log_param("weight_t", params['weight_t'])
            mlflow.log_param("weight_x", params['weight_x'])

        mlflow.log_artifact(params['save_dir']+'visrank_folder_dataset')
        for f in tensorboard_files:
            mlflow.log_artifact(str(f), "tensorboard/")

        mlflow.log_metric("top 1 accuracy", top1_accuracy)
        mlflow.log_metric("mAP", mAP)
        mlflow.log_metric("average cosine distance", av_cosine)
        mlflow.log_metric("max cosine distance", max_cosine)
        mlflow.log_metric("min cosine distance", min_cosine)


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='model_training/train_config.yml',  help='Path to config.yml')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    expirement_with_train(args.config)
