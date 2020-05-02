from sklearn.neighbors import NearestNeighbors
import pandas as pd
from os import getcwd
from os.path import join


def find_nearest_neighbor(df_path, idx = 0):
    """
    Given a datafram path (index must be "problem") and the index of the problem to inspect, will find the problem's
    nearest neighbor and return its name.
    :param df_path: the path of the meta features file
    :param idx: the index of the problem that is inspected
    :return: the name of the nearest neighbor.
    """
    X = pd.read_csv(df_path, index_col='problem')
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    df = pd.read_csv(join(getcwd(), 'meta_data', 'meta_features_all.csv'))

    #  get the index of the nearest neighbor
    most_sim_idx = indices[idx][1]

    return df.iloc[most_sim_idx]['problem']  # returns the  nearest neighbor's name (problem name)

x = find_nearest_neighbor(join(getcwd(), 'meta_data', 'meta_features_all.csv'))
print(x)