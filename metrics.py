from sklearn.metrics import f1_score
import numpy as np


def compute_f1_score(predicted_list: list, target_list: list) -> float:
    """
    Given the lists of target and predicted sequences, it returns the F1-Score

    :param predicted_list: list of predicted sequence
    :param target_list: list of target sequence
    :return: f1_score
    """

    scores = []
    for predicted, target in zip(predicted_list, target_list):
        predicted = [w.strip() for w in predicted.split('[SEP]')]
        target = [w.strip() for w in target.split('[SEP]')]

        # let target and predicted sequences with the same size
        diff_len = len(predicted) - len(target)
        if diff_len > 0:
            target += diff_len * [""]
        elif diff_len < 0:
            predicted += abs(diff_len) * [""]

        scores.append(f1_score(target, predicted, average='macro'))

    return np.array(scores).mean()
