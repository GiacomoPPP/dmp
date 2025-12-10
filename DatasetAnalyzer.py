from math import sqrt
from DmpConfig import DmpConfig
from DatasetGenerator import DatasetGenerator

from torch_geometric.data import Data as Graph

from DmpDataset import DmpDataset

class DatasetAnalyzer:

    def __call__(self, dataset: DmpDataset):

        config = DmpConfig()

        datasetGenerator = DatasetGenerator()

        data, _ = datasetGenerator.get_dataset(dataset, False)

        n_samples = len(data)

        target_list = [graph.y for graph in data]

        max_target = max(target_list)
        min_target = min(target_list)
        avg_target = sum(target_list) / len(target_list)
        variance_target = sum((x - avg_target) ** 2 for x in target_list) / len(target_list)

        print(f"Dataset {config.dataset}")

        print(f"\nAverage target: {avg_target:.2f} \nVariance: {variance_target:.2f} \n Max target: {max_target} \nMin target: {min_target}")

        size_list = [graph.x.shape[0] for graph in data]

        max_size = max(size_list)
        min_size = min(size_list)
        avg_size = sum(size_list) / len(size_list)
        variance_size = sum((x - avg_size) ** 2 for x in size_list) / len(size_list)

        print(f"\nAverage size: {avg_size:.2f} \nVariance: {variance_size:.2f} \n Max size: {max_size} \nMin size: {min_size}")

        edge_list = [graph.edge_index.shape[1] for graph in data]

        avg_edges = sum(edge_list) / len(edge_list)
        stdev_edges = sqrt(sum((x - avg_edges) ** 2 for x in edge_list) / len(edge_list))

        print(rf" {dataset.value} & {n_samples} & {avg_target:.2f} {{\, \scriptstyle \pm {sqrt(variance_target):.2f}}} & {min_target:.2f} & {max_target:.2f} & {avg_size:.0f} {{\, \scriptstyle \pm {sqrt(variance_size):.0f}}} & {min_size} & {max_size} & {avg_edges:.2f} {{\, \scriptstyle \pm {stdev_edges:.2f}}} \\")

if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    for dataset in DmpDataset:
        analyzer(dataset)