import numpy as np
from tqdm import tqdm
import randomMatrix


class HashTable:
    def __init__(self, hash_size, input_dim, data_dict):
        """
        Create a hash table for LSH
        :param hash_size: depth of the hash table
        :param input_dim: input_dim of the data
        :param data_dict: dictionary with training, validation and test data
        """
        np.seed = 42
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.hash_table = {}
        self.data_dict = data_dict
        self.random_matrix = randomMatrix.random_matrix(input_dim, hash_size)
        self.update_hash_table()

    def hash(self, input_vector):
        """
        Compute hash values for the input vector
        :param input_vector: features to be hashed
        :return: embedding based on hash function
        """
        return [int((input_vector @ self.random_matrix[:, i] > 0)) for i in range(self.hash_size)]

    def update_hash_table(self):
        """
        Fill the hash table with the training data
        """
        data, _ = self.data_dict['train']

        for index in data.index:
            track = data.loc[index]
            hash_value = self.hash(track)
            if str(hash_value) not in self.hash_table:
                self.hash_table[str(hash_value)] = []
            self.hash_table[str(hash_value)].append(index)

    def query(self, query_vector):
        """
        Query the hash table for similar tracks
        :param query_vector: features to be queried
        :return: tracks in the same bin as query_vector
        """
        hash_value = self.hash(query_vector)

        if str(hash_value) in self.hash_table:
            return self.hash_table[str(hash_value)]
        else:
            return []


class LSH:
    def __init__(self, hash_size, input_dim, data_dict, num_tables=3, num_tracks=100):
        """
        Create a Locality Sensitive Hashing object
        :param hash_size: depth of the hash table
        :param input_dim: input_dim of the data
        :param data_dict: dictionary with training, validation and test data
        :param num_tables: number of hash tables
        :param num_tracks: number of tracks to consider for prediction
        """
        np.seed = 42
        self.hash_size = hash_size
        self.input_dim = input_dim
        self.num_tracks = num_tracks
        self.data_dict = data_dict
        self.hash_tables = [HashTable(hash_size, input_dim, data_dict) for _ in range(num_tables)]
        self.predictions = {}
        self.genre2idx = {'Hip-Hop': 0, 'Pop': 1, 'Folk': 2, 'Rock': 3, 'Experimental': 4, 'International': 5,
                          'Electronic': 6, 'Instrumental': 7}

    def set_num_tracks(self, num_tracks):
        """
        Set the number of tracks to consider for prediction
        included to streamline the hyperparameter tuning
        :param num_tracks: number of tracks to consider for prediction
        """
        self.num_tracks = num_tracks

    def process(self, evaluation=False):
        """
        Process the validation or test set, return accuracies and confusion matrices
        Assigns the majority genre "Rock" if no similar tracks are found
        :param evaluation: whether to evaluate on the test set or the validation set
        :return: accuracies evaluated with cosine similarity and Euclidean distance and confusion matrices
        """
        random_predictions = {}
        if evaluation:
            features, df = self.data_dict['test']
        else:
            features, df = self.data_dict['validation']
        train_features, df_train = self.data_dict['train']
        prediction = {}
        for index in tqdm(features.index):
            track = features.loc[index]
            similar_tracks = []
            for table in self.hash_tables:
                similar_tracks += table.query(track)
            samples = [train_features.loc[index] for index in similar_tracks]

            cosine_similarities = [self.cosine_similarity(track, sample) for sample in samples]
            euclidean_distances = [self.euclidean_distance(track, sample) for sample in samples]

            try:
                pred_cos = np.argsort(cosine_similarities, axis=0)[-self.num_tracks:]
                cos_indices = [similar_tracks[i] for i in pred_cos]
                pred_cos = df_train.loc[cos_indices]['track']['genre_top']
                pred_cos = pred_cos.value_counts().idxmax()

                pred_euc = np.argsort(euclidean_distances, axis=0)[:self.num_tracks]
                euc_indices = [similar_tracks[i] for i in pred_euc]
                pred_euc = df_train.loc[euc_indices]['track']['genre_top']
                pred_euc = pred_euc.value_counts().idxmax()

            except ValueError:
                pred_euc = 'Rock'
                pred_cos = pred_euc
                if pred_euc not in random_predictions:
                    random_predictions[pred_euc] = 0
                random_predictions[pred_euc] += 1
            prediction[index] = (pred_cos, pred_euc)

        self.predictions = prediction
        cosine_accuracy = 0
        euclidean_accuracy = 0

        genre_dict = self.similarity_per_genre(prediction, evaluation)

        for track_id, (cos, euc) in prediction.items():
            if cos == df.loc[track_id]['track']['genre_top']:
                cosine_accuracy += 1
            if euc == df.loc[track_id]['track']['genre_top']:
                euclidean_accuracy += 1

        cosine_accuracy = round(cosine_accuracy / len(prediction), 3)
        euclidean_accuracy = round(euclidean_accuracy / len(prediction), 3)

        return cosine_accuracy, euclidean_accuracy, genre_dict, *self.confusion_matrix(evaluation)

    def similarity_per_genre(self, prediction_dict, evaluation=False):
        """
        Compute the similarity per genre
        :param evaluation: whether to evaluate on the test set or the validation set
        :param prediction_dict: dictionary of predictions
        :return:
        """
        if evaluation:
            df = self.data_dict['test'][1]
        else:
            df = self.data_dict['validation'][1]
        genre_dict = {}
        for index in prediction_dict:
            track = df.loc[index]
            genre = track['track']['genre_top']
            if genre not in genre_dict:
                genre_dict[genre] = {'Accuracy using cosine similarity': 0.0,
                                     'Accuracy using Euclidean similarity': 0.0, 'Number of samples per Genre': 0.0}
            genre_dict[genre]['Number of samples per Genre'] += 1
            cos, euc = prediction_dict[index]
            if cos == genre:
                genre_dict[genre]['Accuracy using cosine similarity'] += 1.0
            if euc == genre:
                genre_dict[genre]['Accuracy using Euclidean similarity'] += 1.0
        for genre in genre_dict:
            genre_dict[genre]['Accuracy using cosine similarity'] = round(genre_dict[genre][
                                                                              'Accuracy using cosine similarity'] /
                                                                          genre_dict[genre][
                                                                              'Number of samples per Genre'], 3)
            genre_dict[genre]['Accuracy using Euclidean similarity'] = round(genre_dict[genre][
                                                                                 'Accuracy using Euclidean similarity'] /
                                                                             genre_dict[genre][
                                                                                 'Number of samples per Genre'], 3)
        return genre_dict

    def cosine_similarity(self, a, b):
        """
        Compute the cosine similarity between two vectors
        :param a: first input vector
        :param b: second input vector
        :return: cosine similarity
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def euclidean_distance(self, a, b):
        """
        Compute the Euclidean distance between two vectors
        :param a: first input vector
        :param b: second input vector
        :return: Euclidean distance
        """
        return np.linalg.norm(a - b)

    def eval(self):
        """
        Evaluate the test set with the best parameters found during training
        :return: accuracies achieved on test data for both metrics, as well as the per genre accuracies
        """
        results = self.process(evaluation=True)
        return results[:3]

    def confusion_matrix(self, evaluation=False):
        """
        Compute the confusion matrix
        :return: confusion matrix for predictions based on cosine similarity and Euclidean distance
        """
        confusion_matrix_cos = np.zeros((8, 8))
        confusion_matrix_euc = np.zeros((8, 8))
        if evaluation:
            df = self.data_dict['test'][1]
        else:
            df = self.data_dict['validation'][1]
        for index in self.predictions:
            track = df.loc[index]
            genre = track['track']['genre_top']
            cos, euc = self.predictions[index]
            confusion_matrix_cos[self.genre2idx[genre]][self.genre2idx[cos]] += 1
            confusion_matrix_euc[self.genre2idx[genre]][self.genre2idx[euc]] += 1
        return confusion_matrix_cos, confusion_matrix_euc
