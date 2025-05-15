import matplotlib.pyplot as plt

from DatasetGenerator import DatasetGenerator
from DmiConfig import DmiConfig

class Analysis:

    def __init__(self) -> None:
        self.datasetGenerator: DatasetGenerator = DatasetGenerator()


    def analyze_dataset_generator_distribution(self) -> None:

        config = DmiConfig()

        histogram_bins: int = 20

        split1, split2, split3 = self.datasetGenerator.get_dataset(config)

        target1, target2, target3 = self._load_targets(split1, split2, split3)

        self._plot_histograms(histogram_bins, target1, target2, target3)


    def _load_targets(self, *split_list) -> tuple[list, ...]:

        result_list: list = []

        for split in split_list:
            result_list.append(self._extract_target(split))

        return tuple(result_list)


    def _extract_target(self, split: list) -> list:
        return [sample.y for sample in split]


    def _plot_histograms(self, bins: int, *dataset_list) -> None:
        fig, axs = plt.subplots(len(dataset_list), 1, sharex = True)
        fig.suptitle(self._build_histogram_title(dataset_list))
        fig.tight_layout()
        for index, dataset in enumerate(dataset_list):
            axs[index].hist(dataset, bins = bins)
            axs[index].set_title(f"{len(dataset):,} samples")
        plt.show()

    def _build_histogram_title(self, dataset_list) -> str:
        total_samples: int = sum([len(dataset) for dataset in dataset_list])
        return f"Total samples: {total_samples:,}"


if __name__ == "__main__":
    analysis = Analysis()

    analysis.analyze_dataset_generator_distribution()
