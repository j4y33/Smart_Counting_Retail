import json
import numpy as np
import base64
import cv2
import typing as typ
from collections import defaultdict
import os
import json
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
import datetime as dt
import pandas as pd
import pytz
from reader import Reader

from shapely.geometry import Point, Polygon


def in_zone(box, camera):
    if camera == 'entrance':
        return True
    # [700,100, 1350, 500]
    # [1000,0, 2450, 350]
    boxes = {
        'entrance': Polygon([(700, 100), (1350, 100), (1350, 500), (700, 500)]),
        'cash-desk': Polygon([(1100, 250), (1800, 300), (1150, 1700), (50, 1600)])
    }

    polygon = boxes[camera]
    x = int((box[0] + box[2]) / 2)
    y = int((box[1] + box[3]) / 2)
    p = Point(x, y)

    return p.within(polygon)


def time_fps(file_name, t_l, t_h, tz):
    start_datetime = dt.datetime.fromtimestamp(int(file_name.split('_')[0]), tz=tz)
    return t_l <= start_datetime.hour < t_h


def cosine_distance(x, y):
    """
    Arguments:
        x: a numpy float array with shape [n, c].
        y: a numpy float array with shape [m, c].
    Returns:
        a numpy float array with shape [n, m].
    """
    epsilon = 1e-8

    x_norm = np.sqrt((x ** 2).sum(1, keepdims=True))
    y_norm = np.sqrt((y ** 2).sum(1, keepdims=True))

    x = x / (x_norm + epsilon)
    y = y / (y_norm + epsilon)

    product = np.expand_dims(x, 1) * y  # shape [n, m, c]
    cos = product.sum(2)  # shape [n, m]
    return 1.0 - cos


def get_embeddings_from_track(track):
    """Get All embeddings from track"""
    return [pt['embedding'] for pt in track['trajectory'] if isinstance(pt['embedding'], list)]


def get_track_length(track):
    """Get track length"""
    return track['len']


def get_pairwise_distances(tracks):
    pairwise_distances = np.zeros((len(tracks), len(tracks)), dtype=np.float64)
    for i, track_e in enumerate(tracks):
        embeddings_e = np.array(get_embeddings_from_track(track_e), dtype=np.float64)
        for j in range(i + 1, len(tracks)):
            track_q = tracks[j]

            embeddings_q = np.array(get_embeddings_from_track(track_q), dtype=np.float64)

            cos_dists = cosine_distance(embeddings_e, embeddings_q)

            pairwise_distances[i, j] = np.median(cos_dists)
            pairwise_distances[j, i] = pairwise_distances[i, j]
    return pairwise_distances


def save_time_in_store(data_folder, tracks_file, reports_folder, embeddings_limit, tz, camera, min_len):
    df = Reader.get_all_info(data_folder, tracks_file, embeddings_limit)
    videos = sorted(set(df.source))
    groups10 = [videos[n:n + 11] for n in range(0, len(videos), 5)]

    max_l = 0
    reports = pd.DataFrame(columns=['in_time', 'duration', 'cluster_uuid', 'smart_counter_uuid'])
    for group in tqdm(groups10):
        data = df[[x in group for x in df.source]]
        data = list(data.T.to_dict().values())

        tracks = list(filter(lambda x: (len(x['trajectory']) > min_len) and (in_zone(x['first_box'], camera)), data))
        if len(tracks) < 2:
            continue
        
        pairwise_distances = get_pairwise_distances(tracks)

        db = AgglomerativeClustering(affinity='precomputed', linkage='average', n_clusters=None,
                                     distance_threshold=0.49)

        labels = db.fit_predict(X=pairwise_distances)

        unique_labels, unique_counts = np.unique(labels[(labels >= 0)], return_counts=True)
        track_ids = np.array(sorted([x['cluster_id'] for x in data]))
        for label in unique_labels:
            tids = track_ids[labels == label]
            trks = [list(filter(lambda x: x['cluster_id'] == tid, data))[0] for tid in tids]

            if len(trks) != 0:

                st_time = dt.datetime.strptime(trks[0]['start'].split('+')[0].split('.')[0], '%Y-%m-%dT%H:%M:%S')
                end_time = dt.datetime.strptime(trks[-1]['end'].split('+')[0].split('.')[0], '%Y-%m-%dT%H:%M:%S')

                reports = reports.append({'in_time': tz.localize(st_time).isoformat(timespec='microseconds'),
                                          'duration': pd.Timedelta(end_time - st_time).isoformat(),
                                          'cluster_uuid': label + max_l,
                                          'smart_counter_uuid': 1},
                                         ignore_index=True)
        max_l += max(unique_labels)

    reports = reports.sort_values(['in_time'])
    reports.to_csv(reports_folder + '/TiS_events.csv', index=False)


if __name__ == "__main__":
    from shutil import copyfile

    data_folder = 'res/20201010/tracks_data/'
    tracks_file = 'res/20201010/tracks.json'
    reports_folder = 'res/20201010/'
    embeddings_limit = 100
    tz = pytz.timezone('Europe/Paris')
    camera = 'entrance'
    min_len = 3
    save_time_in_store(data_folder, tracks_file,reports_folder, embeddings_limit, tz, camera, min_len)
    # copyfile('res/20210119' + '/time_in_store.csv', '/mnt/smart-counter/entrance/chausse-dantin-1/20210119'+'/time_in_store.csv')
    # copyfile('res/20210119' + '/TiS_events.csv', '/mnt/smart-counter/entrance/chausse-dantin-1/20210119'+'/TiS_events.csv')
