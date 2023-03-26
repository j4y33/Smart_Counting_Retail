import shutil
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import json


def agglo_clustering_window(root: str,
                            new_folder: str,
                            window_minutes: int,
                            distance_threshold: float):
    """
    make crops clusterization with agglomerative algo

    Parameters
    ----------
    root
        path to crops folder
    new_folder
        path to folder with result
    window_minutes
    distance_threshold
    """
    root = Path(root)
    new_folder = Path(new_folder)

    dfs = []
    for folder in root.glob('*'):
        df = pd.read_json(folder/'embeddings.json', lines=True)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.num = df.image_id.map(lambda x: int(x.split('_')[0]))

    last = int(df.iloc[-1].image_id.split('_')[0])
    diff = window_minutes*60

    for i in range(0, last, diff):
        part_df = df[(df.num <= i+diff) & (i <= df.num)]
        part_df = part_df.reset_index()

        nump = part_df.embedding.to_numpy()
        embeddings_track = np.array([np.array(x) for x in nump])

        clustering = AgglomerativeClustering(n_clusters=None,
                                             affinity='cosine',
                                             distance_threshold=distance_threshold,
                                             linkage='average').fit_predict(embeddings_track)

        for ind, cluster in enumerate(clustering):
            cluster = new_folder/str(i)/str(cluster)
            cluster.mkdir(parents=True, exist_ok=True)
            img_name = part_df.loc[ind].image_id
            for img in root.rglob(img_name+'*'):
                shutil.copy(img, cluster/img.name)

                with open(cluster / 'embeddings.json', 'a') as f:
                    json_info = {
                        'image_id': img.stem,
                        'embedding': embeddings_track[ind].tolist()
                    }
                    f.write(json.dumps(json_info)+'\n')

        # for folder in new_folder.glob('*'):
        #     for track in folder.glob('*'):
        #         if len(list(track.glob('*'))) <= 3:
        #             shutil.rmtree(str(track))


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, required=False,
                        default='model_training/crops/crops_all',  help='Path to root')
    parser.add_argument('--new_folder', type=str, required=False,
                        default='model_training/crops/crops_random_merged',  help='Path to res folder')
    parser.add_argument('--window_minutes', type=int, required=False,
                        default=30,  help='window for clustering in minutes')
    parser.add_argument('--distance_threshold', type=float, required=False,
                        default=0.35,  help='cosine distance threshold for clustering')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    agglo_clustering_window(args.root, args.new_folder, args.window_minutes, args.distance_threshold)
