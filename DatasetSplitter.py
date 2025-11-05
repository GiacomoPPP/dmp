import math
from typing import Literal

import numpy as np
from numpy import ndarray

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import KBinsDiscretizer


class DatasetSplitter:
    def __call__(self, data: ndarray, seed: int, split_type: Literal["standard", "stratified"], *split_list) -> list[list]:
        n_bins: int = int(math.sqrt(len(data)))

        bin_membership: ndarray = self._get_bin_membership(data, n_bins)

        if split_type == "standard":
            return self._standard_split(data, seed, split_list)
        elif split_type == "stratified":
            return self._stratified_multiple_split(bin_membership, split_list, seed)


    def _get_bin_membership(self, target_list: ndarray, n_bins: int) -> ndarray:
        formatted_target_list: ndarray = target_list.reshape(-1, 1)

        discretizer = KBinsDiscretizer(n_bins, strategy = 'quantile', encode = 'ordinal')

        discretizer.fit(formatted_target_list)

        bin_membership: ndarray = discretizer.transform(formatted_target_list)

        return bin_membership.squeeze(1)


    def _stratified_multiple_split(self, data: ndarray, split_list: list, seed: int) -> list:

        normalized_split_list = self._normalize_split_list(split_list)

        used_splits: float = 0
        is_available = np.full(len(data), True)
        index_list = np.arange(len(data))

        return_values = []

        for split_size in normalized_split_list:
            relative_split = split_size / (1 - used_splits)

            if abs(relative_split - 1) < 1e-5:
                return_values.append(index_list[is_available])
                break

            current_split = self._stratified_binary_split(relative_split, index_list[is_available], data[is_available], seed)

            return_values.append(current_split)
            is_available[current_split] = False
            used_splits += split_size

        return return_values


    def _stratified_binary_split(self, split_ratio: float, index_list: ndarray, data: ndarray, seed: int) -> ndarray:
        try:
            stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits = 1, train_size = split_ratio, random_state = seed)
            splitter = stratifiedShuffleSplit.split(index_list, data)
            split1, _ = next(splitter)
            return index_list[split1]
        except ValueError:
            print(f"Could not perform stratified splitting on {data}")
            split1, _, _, _ = train_test_split(index_list, data, train_size=split_ratio, shuffle = True, random_state = seed)
            return split1


    def _normalize_split_list(self, split_list: list) -> list:
        normalized_split_list = np.array(split_list, dtype=float)
        normalized_split_list /= normalized_split_list.sum()
        return normalized_split_list

    def _standard_split(self, data: ndarray, seed: int, split_list: tuple[float]):
        rng = np.random.default_rng(seed)
        n = len(data)
        indices = rng.permutation(n)

        split_array = self._normalize_split_list(split_list)

        cut_points = np.round(np.cumsum(split_array) * n).astype(int)
        split_masks = np.split(indices, cut_points[:-1])

        return split_masks
