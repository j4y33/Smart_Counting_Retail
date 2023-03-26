# File with unused clustering algo

def get_embeddings_all_from_file(track_path):
    df = pd.read_json(track_path/'embeddings.json', lines=True)
    nump = df.embedding.to_numpy()
    nump = np.array([np.array(x) for x in nump])
    return nump


def get_embeddings_all_from_folder(folder_path):
    dfs = []  # an empty list to store the data frames
    for folder in folder_path.glob('*'):
        df = pd.read_json(folder/'embeddings.json', lines=True)
        dfs.append(df)  # append the data frame to the list
    df = pd.concat(dfs, ignore_index=True)  # concatenate all the data frames in the list.

    nump = df.embedding.to_numpy()
    nump = np.array([np.array(x) for x in nump])
    return nump


def aglomerative_clustering_crops(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    embeddings_track = get_embeddings_all_from_file(root/'embeddings.json')
    print(embeddings_track.shape)
    clustering = AgglomerativeClustering(n_clusters=None,
                                         affinity='cosine',
                                         distance_threshold=0.31,
                                         linkage='average').fit_predict(embeddings_track)
    print(len(np.unique(clustering)))

    df = pd.read_json(root/'embeddings.json', lines=True)

    #new_folder.mkdir(parents=True, exist_ok=True)
    for ind, cluster in enumerate(clustering):
        (new_folder/str(cluster)).mkdir(parents=True, exist_ok=True)
        img_name = df.loc[ind].image_id
        for img in root.rglob(img_name+'*'):
            shutil.copy(img, new_folder/str(cluster)/img.name)


def aglomerative_clustering_tracks(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    embeddings_track = get_embeddings_all_from_folder(root)
    print(embeddings_track.shape)
    clustering = AgglomerativeClustering(n_clusters=None,
                                         affinity='cosine',
                                         distance_threshold=0.35,
                                         linkage='average').fit_predict(embeddings_track)

    print(len(np.unique(clustering)))

    dfs = []  # an empty list to store the data frames
    for folder in root.glob('*'):
        df = pd.read_json(folder/'embeddings.json', lines=True)
        dfs.append(df)  # append the data frame to the list
    df = pd.concat(dfs, ignore_index=True)  # concatenate all the data frames in the list.

    #new_folder.mkdir(parents=True, exist_ok=True)
    for ind, cluster in enumerate(clustering):
        (new_folder/str(cluster)).mkdir(parents=True, exist_ok=True)
        img_name = df.loc[ind].image_id
        for img in root.rglob(img_name+'*'):
            shutil.copy(img, new_folder/str(cluster)/img.name)

    for folder in new_folder.glob('*'):
        if len(list(folder.glob('*'))) <= 3:
            shutil.rmtree(str(folder))


def random_merged(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    dfs = []  # an empty list to store the data frames

    folders = list(root.glob('*'))
    shuffle(folders)

    folders_normal = folders[len(folders)//2:]
    folders_to_merge = folders[:len(folders)//2]

    for folder in tqdm(folders_normal):
        copy_tree(str(folder), str(new_folder/folder.name))

    print(len(folders_normal))
    print(len(folders_to_merge))

    list_of_m = zip(*(iter(folders_to_merge),) * 2)

    for folder in tqdm(list_of_m):
        name = folder[0].name + '_' + folder[1].name
        copy_tree(str(folder[0]), str(new_folder/name))
        copy_tree(str(folder[1]), str(new_folder/name))

    for f in new_folder.glob('*'):
        print(f)
        try:
            (f/'embeddings.json').unlink()
        except Exception:
            pass


def DBSCAN_method_all_crops(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    embeddings_track = get_embeddings_all_from_file(root/'embeddings.json')
    clustering = DBSCAN(eps=0.139, metric='cosine', min_samples=4).fit_predict(embeddings_track)

    print(len(np.unique(clustering)))

    df = pd.read_json(root/'embeddings.json', lines=True)

    #new_folder.mkdir(parents=True, exist_ok=True)
    for ind, cluster in enumerate(clustering):
        (new_folder/str(cluster)).mkdir(parents=True, exist_ok=True)
        img_name = df.loc[ind].image_id
        for img in root.rglob(img_name+'*'):
            shutil.copy(img, new_folder/str(cluster)/img.name)


# много отсеяли, не уверен, что это не надо тюнить, много не смерджило (плохо мерджит разные позы)


def DBSCAN_method(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    embeddings_track = get_embeddings_all_from_folder(root)
    clustering = DBSCAN(eps=0.139, metric='cosine', min_samples=4).fit_predict(embeddings_track)

    print(len(np.unique(clustering)))

    dfs = []  # an empty list to store the data frames
    for folder in root.glob('*'):
        df = pd.read_json(folder/'embeddings.json', lines=True)
        dfs.append(df)  # append the data frame to the list
    df = pd.concat(dfs, ignore_index=True)  # concatenate all the data frames in the list.

    #new_folder.mkdir(parents=True, exist_ok=True)
    for ind, cluster in enumerate(clustering):
        (new_folder/str(cluster)).mkdir(parents=True, exist_ok=True)
        img_name = df.loc[ind].image_id
        for img in root.rglob(img_name+'*'):
            shutil.copy(img, new_folder/str(cluster)/img.name)


def random_clustered(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    dfs = []  # an empty list to store the data frames
    for folder in root.glob('*'):
        df = pd.read_json(folder/'embeddings.json', lines=True)
        dfs.append(df)  # append the data frame to the list
    df = pd.concat(dfs, ignore_index=True)  # concatenate all the data frames in the list.

    names = df.image_id.values.tolist()
    random.shuffle(names)

    c = 0
    while len(names) > 10:
        (new_folder/str(c)).mkdir(parents=True, exist_ok=True)
        r = random.randint(10, 20)
        n = names[:r]
        names = names[r:]
        for i in n:
            for img in root.rglob(i+'*'):
                shutil.copy(img, new_folder/str(c)/img.name)
        c += 1


def max_clustering(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))
    for ind1, track in enumerate(root.glob("*")):
        if ind1 in list_of_merged:  # or len(list(track.glob('*.jpg'))) < 3:
            continue

        embeddings_track = get_embeddings_all_from_file(track)
        copy_tree(str(track), str(new_folder/track.name))
        for ind2, other_track in enumerate(root.glob('*')):
            if ind2 <= ind1:  # or len(list(other_track.glob('*.jpg'))) < 3:
                continue
            embeddings_track_other = get_embeddings_all_from_file(other_track)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)

            res[ind1][ind2] = res[ind2][ind1] = np.average(distance_matrix)
            if (np.average(distance_matrix) < 0.31):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                print(track.name, f" max {np.average(distance_matrix)} ", other_track.name)


def max_clustering_best(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))
    for ind1, track in enumerate(root.glob("*")):
        if ind1 in list_of_merged:
            continue

        embeddings_track = get_embeddings_all_from_file(track)
        copy_tree(str(track), str(new_folder/track.name))
        for ind2, other_track in enumerate(root.glob('*')):
            if ind2 <= ind1:
                continue
            embeddings_track_other = get_embeddings_all_from_file(other_track)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)
            if (np.min(distance_matrix) < 0.31):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                print(track.name, f" best {np.min(distance_matrix)} ", other_track.name)


# хороший результат, пусть и есть чуть ошибок (0.8), но дольше


def max_clustering_with_remembering(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))

    tracks = list(root.glob("*"))
    ind1 = 0
    while ind1 < len(tracks)-1:
        if ind1 in list_of_merged:
            ind1 += 1
            continue
        track = tracks[ind1]

        embeddings_track = get_embeddings_all_from_file(track)
        copy_tree(str(track), str(new_folder/track.name))

        ind2 = ind1+1
        while ind2 < len(tracks)-1:
            if ind2 in list_of_merged:
                ind2 += 1
                continue
            other_track = tracks[ind2]
            embeddings_track_other = get_embeddings_all_from_file(other_track)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)

            if (np.average(distance_matrix) < 0.31):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                ind1 = -1
                ind2 = 0
                print(track.name, f" max {np.average(distance_matrix)} ", other_track.name)
                break
            ind2 += 1
        ind1 += 1

# оч плохо (0.41)


def maxp_clustering(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))
    for ind1, track in enumerate(root.glob("*")):
        if ind1 in list_of_merged:
            continue

        embeddings_track = get_embeddings_all_from_file(track)
        copy_tree(str(track), str(new_folder/track.name))
        for ind2, other_track in enumerate(root.glob('*')):
            if ind2 <= ind1:
                continue
            embeddings_track_other = get_embeddings_all_from_file(other_track)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)

            res[ind1][ind2] = res[ind2][ind1] = np.average(distance_matrix)
            if (np.average(distance_matrix) < 0.42):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                print(track.name, f" max+ {np.average(distance_matrix)} ", other_track.name)

# идентично max


def mean_clustering(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))
    for ind1, track in enumerate(root.glob("*")):
        if ind1 in list_of_merged:
            continue

        embeddings_track = get_embeddings_all_from_file(track)
        copy_tree(str(track), str(new_folder/track.name))
        for ind2, other_track in enumerate(root.glob('*')):
            if ind2 <= ind1:
                continue
            embeddings_track_other = get_embeddings_all_from_file(other_track)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)

            if (np.mean(distance_matrix) < 0.31):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                print(track.name, f" max {np.mean(distance_matrix)} ", other_track.name)

# лишние совмещаем


def av_emb_max_clustering(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))
    for ind1, track in enumerate(root.glob("*")):
        if ind1 in list_of_merged:
            continue

        embeddings_track = get_embeddings_all_from_file(track)
        embeddings_track = np.expand_dims(np.average(embeddings_track, axis=0), axis=0)

        copy_tree(str(track), str(new_folder/track.name))
        for ind2, other_track in enumerate(root.glob('*')):
            if ind2 <= ind1:
                continue
            embeddings_track_other = get_embeddings_all_from_file(other_track)
            embeddings_track_other = np.expand_dims(np.average(embeddings_track_other, axis=0), axis=0)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)

            if (np.average(distance_matrix) < 0.31):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                print(track.name, f" max {np.mean(distance_matrix)} ", other_track.name)


# хуже совмещает (76 треков)
# mean идентично
def av_emb_av_clustering(root, new_folder):
    root = Path(root)
    new_folder = Path(new_folder)

    list_of_merged = []

    res = np.zeros((len(list(root.glob("*"))), len(list(root.glob("*")))))
    for ind1, track in enumerate(root.glob("*")):
        if ind1 in list_of_merged:
            continue

        embeddings_track = get_embeddings_all_from_file(track)
        embeddings_track = np.expand_dims(np.mean(embeddings_track, axis=0), axis=0)

        copy_tree(str(track), str(new_folder/track.name))
        for ind2, other_track in enumerate(root.glob('*')):
            if ind2 <= ind1:
                continue
            embeddings_track_other = get_embeddings_all_from_file(other_track)
            embeddings_track_other = np.expand_dims(np.mean(embeddings_track_other, axis=0), axis=0)

            distance_matrix = cosine_distance_vectorized(embeddings_track, embeddings_track_other)

            if (np.average(distance_matrix) < 0.139):
                list_of_merged.append(ind2)
                copy_tree(str(other_track), str(new_folder/track.name))
                print(track.name, f" max {np.mean(distance_matrix)} ", other_track.name)


def get_real(start_f, end_f, path):
    path = Path(path)
    df = pd.read_csv(path)
    df['Номер кадра - вход'] = df['Номер кадра - вход']/25
    part_df = df[(start_f <= df['Номер кадра - вход']) & (df['Номер кадра - вход'] <= end_f)]
    part_df = df
    return len(part_df)
