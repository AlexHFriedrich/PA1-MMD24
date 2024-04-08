import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file):
    """
    Load data from csv file
    :param file: path to csv file
    :return: pandas Dataframe
    """
    return pd.read_csv(file, index_col=0, header=[0, 1], low_memory=False)


def prep_data(tracks, features):
    """
    Prepare data for processing
    :param tracks: Dataframe with track information
    :param features: Dataframe with features
    :return: filtered and split dataframes
    """
    df_tracks = tracks[tracks['set']['subset'] == 'medium']
    relevant_genres = ['Hip-Hop', 'Pop', 'Folk', 'Rock', 'Experimental', 'International', 'Electronic', 'Instrumental']
    df_tracks = df_tracks[df_tracks['track']['genre_top'].isin(relevant_genres)]
    df_features = features

    df_train = df_tracks[df_tracks['set']['split'] == 'training']
    df_validation = df_tracks[df_tracks['set']['split'] == 'validation']
    df_test = df_tracks[df_tracks['set']['split'] == 'test']

    return df_train, df_validation, df_test, df_features


def process_data(file_paths):
    """
    Process data for LSH
    :param file_paths: list of file paths
    :return: dictionary with train, validation and test data
    """
    print("Processing data...")
    files = [load_data(file) for file in file_paths]

    df_train, df_validation, df_test, df_features = prep_data(files[0], files[1])

    train_ids = df_train.index
    validation_ids = df_validation.index
    test_ids = df_test.index

    # drop first two rows of df_features
    df_features = df_features.drop(df_features.index[0:2])
    df_features.index = df_features.index.map(int)
    train_features = df_features.loc[train_ids]
    validation_features = df_features.loc[validation_ids]
    test_features = df_features.loc[test_ids]

    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    validation_features = scaler.transform(validation_features)
    test_features = scaler.transform(test_features)

    train_features = pd.DataFrame(train_features, index=train_ids)
    validation_features = pd.DataFrame(validation_features, index=validation_ids)
    test_features = pd.DataFrame(test_features, index=test_ids)

    data_dict = {
        'train': (train_features, df_train),
        'validation': (validation_features, df_validation),
        'test': (test_features, df_test)
    }
    print("Data processing done!")
    return data_dict


if __name__ == "__main__":
    pass
