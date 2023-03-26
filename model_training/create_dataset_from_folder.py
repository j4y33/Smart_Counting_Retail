from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
from argparse import ArgumentParser


def main(res_folder: str,
         crops_folder: str,
         train_test_split: float,
         gallery_query_split: float
         ):
    """
    creates 3 folder (training, query, gallery)
    file naming - {person_id}_c{camera_id}_{image_id}.jpg
    """
    res_folder = Path(res_folder)
    crops_folder = Path(crops_folder)

    counter = 0

    if not crops_folder.exists():
        return

    # get all tracks and split them to train and test
    tracks = list(crops_folder.glob("*"))
    train_tracks, test_tracks = train_test_split(tracks, test_size=train_test_split)

    # create training folder
    training_folder = res_folder/'training'
    training_folder.mkdir(parents=True, exist_ok=True)
    for track in tqdm(train_tracks):
        counter += 1
        for img in track.glob("*.jpg"):
            file_name = training_folder/(str(counter)+'_c1_'+img.name.split('_')[-1])
            shutil.copy(img, file_name)

    # create query and gallery folders
    query_folder = res_folder/'query'
    gallery_folder = res_folder/'gallery'
    query_folder.mkdir(parents=True, exist_ok=True)
    gallery_folder.mkdir(parents=True, exist_ok=True)
    for track in tqdm(test_tracks):
        try:
            counter += 1
            # split images for query and gallery (it`s obligatory for cameras to be different)
            imgs = list(track.glob("*.jpg"))
            imgs_gallery, imgs_query = train_test_split(imgs, test_size=gallery_query_split)
            for img in imgs_query:
                file_name = query_folder/(str(counter)+'_c1_'+img.name.split('_')[-1])
                shutil.copy(img, file_name)
            for img in imgs_gallery:
                file_name = gallery_folder/(str(counter)+'_c2_'+img.name.split('_')[-1])
                shutil.copy(img, file_name)
        except Exception:
            pass


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--res_folder', type=str, required=False,
                        default='model_training/crops/dataset_4_328',  help='Path to res folder')
    parser.add_argument('--crops_folder', type=str, required=False,
                        default='model_training/crops/crops_all',
                        help='Path to folder with crops')
    parser.add_argument('--train_test_split', type=float, required=False, default=0.2,
                        help='percent for test tracks')
    parser.add_argument('--gallery_query_split', type=float, required=False, default=0.15,
                        help='percent for query images')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.res_folder, args.crops_folder, args.train_test_split, args.gallery_query_split)
