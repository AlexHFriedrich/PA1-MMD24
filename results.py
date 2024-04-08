import pandas as pd


def create_results_dict():
    """
    Create a dictionary to store results of training
    :return:
    """
    return {
        'best_cos_acc': 0,
        'best_euc_acc': 0,
        'best_parameters_cos': (0, 0, 0),
        'best_parameters_euc': (0, 0, 0),
        'best_genre_dict': {},
        'best_cosine_performance_time': 0,
        'best_euc_performance_time': 0,
        'cosine_confusion_matrix': None,
        'euclidean_confusion_matrix': None,

    }


def update_results_dict_cosine(results_dict, num_tables, num_hashes, num_tracks, temp_cos, genre_dict,
                               confusion_matrix_cos):
    """
    Update the results dictionary with the best results obtained w.r.t. cosine similarity
    :param results_dict:
    :param num_tables:
    :param num_hashes:
    :param num_tracks:
    :param temp_cos:
    :param genre_dict:
    :param confusion_matrix_cos:
    :return:
    """
    results_dict['best_cos_acc'] = temp_cos
    results_dict['best_parameters_cos'] = (num_tables, num_hashes, num_tracks)
    results_dict['best_genre_dict'] = genre_dict
    results_dict['cosine_confusion_matrix'] = confusion_matrix_cos
    return results_dict


def update_results_dict_euclidean(results_dict, num_tables, num_hashes, num_tracks, temp_euc, genre_dict,
                                  confusion_matrix_euc):
    """
    Update the results dictionary with the best results obtained w.r.t. Euclidean distance
    :param results_dict:
    :param num_tables:
    :param num_hashes:
    :param num_tracks:
    :param temp_euc:
    :param genre_dict:
    :param confusion_matrix_euc:
    :return:
    """
    results_dict['best_euc_acc'] = temp_euc
    results_dict['best_parameters_euc'] = (num_tables, num_hashes, num_tracks)
    results_dict['best_genre_dict'] = genre_dict
    results_dict['euclidean_confusion_matrix'] = confusion_matrix_euc
    return results_dict


def write_results_to_file(results_dict, genre2idx):
    """
    Write the training results to a file
    :param results_dict:
    :param genre2idx:
    :return:
    """
    with open('best_parameters.txt', 'w') as file:
        file.write(str(results_dict['best_parameters_cos']) + "\n")
        file.write(str(results_dict['best_parameters_euc']))
        file.close()

    with (open('result_file', 'w') as file):
        res_str = ("Results for LSH\n\n"
                   f"Best parameters for cosine similarity: {results_dict['best_parameters_cos'][0]} tables, "
                   f"{results_dict['best_parameters_cos'][1]} hashes, {results_dict['best_parameters_cos'][2]} tracks"
                   f"\nBest parameters for euclidean distance: {results_dict['best_parameters_euc'][0]} tables, "
                   f"{results_dict['best_parameters_euc'][1]} hashes, {results_dict['best_parameters_euc'][2]} tracks.\n"
                   f"Best accuracies: {results_dict['best_cos_acc']}, {results_dict['best_euc_acc']}\n")
        genre_df = pd.DataFrame(results_dict['best_genre_dict']).T
        genre_df['Number of samples per Genre'] = genre_df['Number of samples per Genre'].astype(int)
        genre_str = genre_df.to_string()
        res_str += (f"\nGenre-wise accuracies:\n{genre_str}\n"
                    f"\n\n Legend for confusion matrix: {genre2idx}\n"
                    f"Confusion matrix for cosine similarity:\n"
                    f"{results_dict['cosine_confusion_matrix']}"
                    f"\n\nConfusion matrix for euclidean distance:\n {results_dict['euclidean_confusion_matrix']}")

        file.write(res_str)
        file.close()
        print("Results written to file!\n\n")


def write_eval_results(eval_results):
    """
    Write the evaluation results to a file
    :param eval_results:
    :return:
    """
    with open('evaluation_results.txt', 'w') as file:
        for key in eval_results:
            file.writelines(f"Results for {key} similarity:\n")
            acc_cos, acc_euc, genre_dict = eval_results[key]
            if key == "Cosine":
                file.write(f"Accuracy using cosine similarity: {acc_cos}\n")
            else:
                file.write(f"Accuracy using Euclidean distance: {acc_euc}\n")
            genre_df = pd.DataFrame(genre_dict).T
            file.write(genre_df.to_string())
            file.write("\n\n")
        file.close()
        print("Evaluation results written to file!")
