import csv
from math import sqrt
from typing import Dict
from Analyzer import DmpConfig
from DmpDataset import DmpDataset
import DmpModel

import torch

from torch_geometric.data import Data as Graph
from torch_geometric.loader import DataLoader

from lightning import Trainer

config = DmpConfig

class ModelAssessment:

    def __init__(self):
        self.file_path = "results.csv"


    def compute_root_mean_square(self, sample_list: list[Graph]):
        squared_values_list = [graph.y**2 for graph in sample_list]

        mean_square: float = sum(squared_values_list) / len(squared_values_list)

        return sqrt(mean_square)


    def test(self, model: DmpModel, graph_list: list) -> float:
        trainer = Trainer()

        relative_test_loss = trainer.test(model, dataloaders=DataLoader(graph_list))[0].get('test_rmse')

        model.register_buffer("test_rmse", torch.tensor(relative_test_loss))

        return relative_test_loss


    def write_results(self, relative_test_list: Dict[DmpDataset, float]):
        with open(self.file_path, "a") as f:
            writer = csv.writer(f)

            error_list: list[float] = [relative_test_list[dataset] for dataset in DmpDataset]

            truncated_error_list: list[str] = [f"{error:.2f}" for error in error_list]

            writer.writerow([config.model_name] + truncated_error_list)


    def get_saved_model_names(self) -> list[str]:
        model_name_list = []
        with open(self.file_path, newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    model_name_list.append(row[0])

        return model_name_list