from model_training.MarknetBasedDataset import MarknetBased
import torchreid
from argparse import ArgumentParser


def get_ranks(path: str,
              root: str,
              save_dir: str,
              optimizer: str,
              alpha: float,
              width: int = 128,
              height: int = 256,
              batch_size: int = 32):
    """
    Run built-in test on ready model

    Parameters
    ----------
    path
        path to model weights
    root
        dataset root
    save_dir
        path to save ranks
    optimizer
    alpha
    width
    height
    batch_size

    Returns
    -------
    model
    """

    try:
        torchreid.data.register_image_dataset('folder_dataset', MarknetBased)
    except ValueError:
        "alredy initialized"

    datamanager = torchreid.data.ImageDataManager(
        root=root,
        sources='folder_dataset',
        targets='folder_dataset',
        height=height,
        width=width,
        batch_size_train=batch_size,
        batch_size_test=batch_size*2,
        transforms=None
    )

    model = torchreid.models.osnet_ain.osnet_ain_x1_0(pretrained=True)
    torchreid.utils.load_pretrained_weights(model, path)
    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=optimizer,
        lr=alpha
    )
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer
    )

    engine.run(
        save_dir=save_dir,
        dist_metric='cosine',
        visrank=True,
        test_only=True
    )

    return model


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=False,
                        default='model_training/models/resnet50/model/model.pth.tar-60',  help='Path to model weights')
    parser.add_argument('--save_dir', type=str, required=False,
                        default='model_training/models/resnet50',
                        help='Path where to save visranks')
    parser.add_argument('--root', type=str, required=False,
                        default='model_training/crops/dataset/',
                        help='path to dataset')
    parser.add_argument('--optimizer', type=str, required=False,
                        default='adam',
                        help='optimizer name')
    parser.add_argument('--alpha', type=float, required=False,
                        default=0.0003,
                        help='alpha param')
    parser.add_argument('--width', type=int, required=False,
                        default=128,
                        help='img width')
    parser.add_argument('--heigth', type=int, required=False,
                        default=256,
                        help='img heigth')
    parser.add_argument('--batch_size', type=int, required=False,
                        default=32,
                        help='batch_size')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    get_ranks(args.path, args.root, args.save_dir, args.optimizer, args.alpha, args.width, args.height, args.batch_size)
