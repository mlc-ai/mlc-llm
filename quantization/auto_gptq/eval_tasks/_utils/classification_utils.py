import sys
from typing import List, Sequence

import numpy as np


def levenshtein_distance(seq1: Sequence, seq2: Sequence):
    if seq1 == seq2:
        return 0
    num_rows = len(seq1) + 1
    num_cols = len(seq2) + 1
    dp_matrix = np.empty((num_rows, num_cols))
    dp_matrix[0, :] = range(num_cols)
    dp_matrix[:, 0] = range(num_rows)

    for i in range(1, num_rows):
        for j in range(1, num_cols):
            if seq1[i - 1] == seq2[j - 1]:
                dp_matrix[i, j] = dp_matrix[i - 1, j - 1]
            else:
                dp_matrix[i, j] = min(dp_matrix[i - 1, j - 1], dp_matrix[i - 1, j], dp_matrix[i, j - 1]) + 1

    return dp_matrix[num_rows - 1, num_cols - 1]


def get_closest_label(pred: Sequence, classes: List[Sequence]) -> int:
    min_id = sys.maxsize
    min_edit_distance = sys.maxsize
    for i, class_label in enumerate(classes):
        edit_distance = levenshtein_distance(pred, class_label)
        if edit_distance < min_edit_distance:
            min_id = i
            min_edit_distance = edit_distance
    return min_id


__all__ = ["levenshtein_distance", "get_closest_label"]
