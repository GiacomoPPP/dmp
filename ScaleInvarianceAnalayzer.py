from Analyzer import Analyzer
from DatasetGenerator import DatasetGenerator
from DmiConfig import DmiConfig
from DmiModel import DmiModel

import torch

from torch import Tensor

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

import torch.nn.functional as F

import matplotlib.pyplot as plt



config = DmiConfig()

class ScaleInvarianceAnalayzer(Analyzer):

    def __call__(self) -> None:
        model, model_name = self._get_model()

        print(f"Analysis for model {model_name}")

        full_graph_list, scaled_graph_list = self._get_datasets(model.geometric_scale)

        predicted_full_list, actual_full_list, full_error = self.analyze_dataset(model, full_graph_list)

        print(f"Error for full graph: {full_error}")

        predicted_scaled_list, actual_scaled_list, scaled_error = self.analyze_dataset(model, scaled_graph_list)

        print(f"Error for scaled graph: {scaled_error}")

        if(torch.equal(actual_full_list, actual_scaled_list)):
            self._compare_predictions(predicted_full_list, predicted_scaled_list, model_name)
        else:
            pass

    def _get_datasets(self, geometric_scale: float) -> tuple[list[Graph], list[Graph]]:
        dataset_generator = DatasetGenerator()

        full_graph_list, _ = dataset_generator.get_whole_dataset(config, True, geometric_scale)

        scaled_graph_list, _ = dataset_generator.get_whole_dataset(config, False, geometric_scale)

        return full_graph_list, scaled_graph_list

    def analyze_dataset(self, model: DmiModel, graph_list: list[Graph]) -> tuple[Tensor, Tensor, float]:
        prediceted_list: Tensor = self._evaluate(model, graph_list)

        actual_list: Tensor = Tensor([graph.y for graph in graph_list]).to(config.device)

        error: float = F.mse_loss(prediceted_list, actual_list)

        return prediceted_list, actual_list, error


    def _get_model(self) -> tuple[DmiModel, str]:
        model_params, model_name = self._load_model()

        model = DmiModel(config)

        model.load_state_dict(model_params)

        model.eval()

        model.to(config.device)

        return model, model_name


    def _evaluate(self, model: DmiModel, dataset: list[Graph]) -> Tensor:
        loader = DataLoader(dataset, batch_size=len(dataset))

        with torch.no_grad():
            for batch in loader:
                return model(batch.to(config.device))

    def _compare_predictions(self, full_prediction: Tensor, scaled_prediction: Tensor, model_name: str) -> None:


        difference = torch.stack((full_prediction, scaled_prediction), 1)

        print(difference)

        difference: Tensor = torch.abs(full_prediction - scaled_prediction)

        average_difference = difference.mean()

        print(f"Average difference: {average_difference}")

        num_bins = 50
        plt.hist(difference.tolist(), bins=num_bins, edgecolor='black')
        plt.title(f"Differences between full and scaled graphs predictions \n Model {model_name}")
        plt.grid(True)
        plt.show()