import numpy as np
from LSH import LSH
from data_preparation import process_data
from results import create_results_dict, update_results_dict_cosine, update_results_dict_euclidean, \
    write_results_to_file, write_eval_results

if __name__ == "__main__":
    np.seed = 42

    # set this to ['path to tracks.csv', 'path to features.csv']
    file_paths = ['data/fma_metadata/tracks.csv', 'data/fma_metadata/features.csv']
    data_dict = process_data(file_paths)
    input_dim = data_dict['train'][0].shape[1]

    # set this to True to run the hyperparameter tuning
    training = False
    if training:
        parameter_grid = {
            'num_tables': [20, 30, 40],
            'num_hashes': [17, 19, 21],
            'num_tracks': [10, 25, 40]
        }

        results_dict = create_results_dict()

        for num_tables in parameter_grid['num_tables']:
            for num_hashes in parameter_grid['num_hashes']:
                # Initialize LSH object outside the inner loop to improve training time
                lsh = LSH(num_hashes, input_dim, data_dict, num_tables, 10)
                for num_tracks in parameter_grid['num_tracks']:
                    lsh.set_num_tracks(num_tracks)
                    cos_acc, euc_acc, genre_dict, c_m_cos, c_m_euc = lsh.process()

                    if cos_acc > results_dict['best_cos_acc']:
                        update_results_dict_cosine(results_dict, num_tables, num_hashes, num_tracks, cos_acc,
                                                   genre_dict,
                                                   c_m_cos)

                    if euc_acc > results_dict['best_euc_acc']:
                        update_results_dict_euclidean(results_dict, num_tables, num_hashes, num_tracks, euc_acc,
                                                      genre_dict,
                                                      c_m_euc)

        write_results_to_file(results_dict, lsh.genre2idx)

    with open('best_parameters.txt', 'r') as file:
        best_parameters_cos = tuple(map(int, file.readline().strip('(').strip(')\n').split(',')))
        best_parameters_euc = tuple(map(int, file.readline().strip('(').strip(')\n').split(',')))
        file.close()

    print("Evaluating on test set...")
    eval_results = {}
    print("\nRun with best parameters w.r.t cosine similarity:")
    lsh = LSH(best_parameters_cos[1], input_dim, data_dict, best_parameters_cos[0], best_parameters_cos[2])
    eval_results["Cosine"] = lsh.eval()

    print("\nRun with best parameters w.r.t Euclidean distance:")
    lsh = LSH(best_parameters_euc[1], input_dim, data_dict, best_parameters_euc[0], best_parameters_euc[2])
    eval_results["Euclidean"] = lsh.eval()

    write_eval_results(eval_results)
