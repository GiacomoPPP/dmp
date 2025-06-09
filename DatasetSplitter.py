import math

import numpy as np
from numpy import ndarray

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import KBinsDiscretizer


class DatasetSplitter:
    def __call__(self, data: ndarray, seed: int, *split_size_list) -> list[list]:
        n_bins: int = int(math.sqrt(len(data)))

        bin_membership: ndarray = self._get_bin_membership(data, n_bins)

        return self._stratified_multiple_split(bin_membership, split_size_list, seed)


    def _get_bin_membership(self, target_list: ndarray, n_bins: int) -> ndarray:
        formatted_target_list: ndarray = target_list.reshape(-1, 1)

        discretizer = KBinsDiscretizer(n_bins, strategy = 'quantile', encode = 'ordinal')

        discretizer.fit(formatted_target_list)

        bin_membership: ndarray = discretizer.transform(formatted_target_list)

        return bin_membership.squeeze(1)


    def _stratified_multiple_split(self, data: ndarray, split_size_list: list, seed: int) -> list:

        normalized_split_list = np.array(split_size_list, dtype=float)
        normalized_split_list /= normalized_split_list.sum()

        return_values = []
        available_data: ndarray = data
        consumed_data: float = 0
        available_index_list = np.arange(len(data))

        for split_ratio in normalized_split_list:
            # Compute the ratio of the current split on the sample that are still available
            relative_split_ratio = split_ratio / (1 - consumed_data)

            if abs(relative_split_ratio - 1) < 1e-5:
                return_values.append(available_index_list.tolist())
                break

            split, remaining_data = self._stratified_binary_split(relative_split_ratio, available_index_list, available_data, seed)

            return_values.append(split.tolist())

            available_data = available_data[remaining_data]
            consumed_data += split_ratio
            available_index_list = np.delete(available_index_list, split)

        return return_values


    def _stratified_binary_split(self, split_ratio: float, X: ndarray, y: ndarray, seed: int) -> tuple[ndarray, ndarray]:
            try:
                stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits = 1, train_size = split_ratio, random_state = seed)
                splitter = stratifiedShuffleSplit.split(X, y)
                return next(splitter)
            except ValueError:
                _, _, split, remaining = train_test_split(X, np.arange(len(y)), train_size=split_ratio, random_state=42)
                return split, remaining


    def _remove_values(self, values_list: ndarray, values_to_remove_list: ndarray) -> ndarray:
        return np.array([value for value in values_list if value not in values_to_remove_list])
