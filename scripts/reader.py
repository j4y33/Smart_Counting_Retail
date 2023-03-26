import os
import json
import pandas as pd


class Reader():

    @staticmethod
    def get_main_info(tracks):
        res = []
        try:
            with open(tracks, 'r') as f:
                for line in f:
                    l = json.loads(line)
                    l['first_box'] = l['trajectory'][0]['bounding_box']
                    l['last_box'] = l['trajectory'][-1]['bounding_box']
                    res.append(l)
        except IOError:
            pass
        return pd.DataFrame(res)

    @staticmethod
    def get_all_info(data, tracks, embeddings_limit):
        res = []
        try:
            with open(tracks, 'r') as f:
                for line in f:
                    l = json.loads(line)
                    my_trajectory = []
                    with open(data + str(l['track_id']) + '.json', 'r') as ftr:
                        al = ftr.readlines()
                        diff = len(al)//embeddings_limit
                        diff = 1 if diff == 0 else diff
                        for ind, el in enumerate(al):
                            if (ind % diff) == 0:
                                my_trajectory.append(l['trajectory'][ind])
                                my_trajectory[-1].update(json.loads(el))
                    l['trajectory'] = my_trajectory
                    l['first_box'] = l['trajectory'][0]['bounding_box']
                    l['last_box'] = l['trajectory'][-1]['bounding_box']
                    res.append(l)
        except IOError:
            pass
        return pd.DataFrame(res)


# r = Reader('res/20210117')
# res = r.get_main_info()
# print(res)
