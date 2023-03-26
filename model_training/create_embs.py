import json
from torchreid.utils.feature_extractor import FeatureExtractor
from torchvision.transforms.transforms import Compose, Normalize, Resize, ToTensor
from model_training.get_metrics import get_embeddings_all
from pathlib import Path
from tqdm import tqdm


def create_embs(root: str,
                model_name: str,
                weights: str,
                device: str,
                width: int,
                height: int):
    """
    create embeddings json file fot clusters
    """

    _MEAN = [0.485, 0.456, 0.406]
    _STD = [0.229, 0.224, 0.225]
    root = Path(root)

    model = FeatureExtractor(
        model_name=model_name,
        model_path=weights,
        device=device
    )
    transform = Compose([
        Resize((height, width)),
        ToTensor(),
        Normalize(mean=_MEAN, std=_STD),
    ])

    for folder in tqdm(root.glob('*')):
        if (folder/'embeddings.json').exists():
            continue

        for img in folder.glob('*.jpg'):
            name = img.stem
            emb = get_embeddings_all([img], model, 'cuda:0', transform)
            with open(folder/'embeddings.json', 'a') as f:
                json_info = {
                    'image_id': name,
                    'embedding': emb[0].tolist()
                }
                f.write(json.dumps(json_info)+'\n')


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=False,
                        default='/home/darklen/workspace/smart-counter-reid/model_training/crops/crops_random_merged',
                        help='Path to folder with clusters')
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
    create_embs(args.root, args.model_name, args.weights, args.device, args.width, args.height)
