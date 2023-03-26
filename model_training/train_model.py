from model_training.MarknetBasedDataset import MarknetBased
import torchreid
import yaml
from argparse import ArgumentParser


def train_model(root: str,
                width: int,
                height: int,
                batch_size: int,
                engine: str,
                model: str,
                optimizer: str,
                alpha: float,
                stepsize: int,
                save_dir: str,
                max_epoch: int,
                margin: float = None,
                weight_t: float = None,
                weight_x: float = None,
                transforms: [str] = None):

    try:
        torchreid.data.register_image_dataset('folder_dataset', MarknetBased)
    except ValueError:
        "alredy initialized"

    train_sampler = 'RandomIdentitySampler' if engine == 'ImageTripletEngine' else 'RandomSampler'
    datamanager = torchreid.data.ImageDataManager(
        root=root,
        sources='folder_dataset',
        height=height,
        width=width,
        batch_size_train=batch_size,
        batch_size_test=batch_size*2,
        transforms=transforms,
        train_sampler=train_sampler
    )

    loss = 'triplet' if engine == 'ImageTripletEngine' else 'softmax'
    model = torchreid.models.osnet_ain.osnet_ain_x1_0(pretrained=True, loss=loss) if model == 'osnet_ain_x1_0' else None

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=optimizer,
        lr=alpha
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=stepsize
    )

    engine = torchreid.engine.ImageTripletEngine(
        datamanager, model, optimizer, margin=margin,
        weight_t=weight_t, weight_x=weight_x, scheduler=scheduler
    ) if engine == 'ImageTripletEngine' else torchreid.engine.ImageSoftmaxEngine(
        datamanager, model, optimizer, scheduler=scheduler
    )

    engine.run(
        save_dir=save_dir,
        max_epoch=max_epoch,
        dist_metric='cosine',
        eval_freq=10,
        print_freq=10,
        test_only=False
    )


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, required=False,
                        default='model_training/train_config.yml',  help='Path to config.yml')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    params = yaml.safe_load(open(args.config, 'r'))

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
