import os

from torch_geometric.loader import DataLoader

import torch
from torch.nn import MSELoss
from torch.optim import SGD
import torch.nn as nn

from lightning import Trainer

from DatasetGenerator import DatasetGenerator
from config import EctConfig

from model import EctCnnModel


def get_model_setup(graph_batch: list, config: EctConfig) -> tuple[EctCnnModel, DataLoader, SGD, MSELoss] :
    model = EctCnnModel(config)

    loader = DataLoader(graph_batch, config.n_minibatch, shuffle = True)

    trainer = Trainer(limit_train_batches = config.n_minibatch, max_epochs = config.n_epochs)

    return model, loader, trainer


def train(graph_batch: list, config: EctConfig) -> nn.Module:

    model, loader, trainer = get_model_setup(graph_batch, config)

    trainer.fit(model, train_dataloaders = loader)

    return model

def test(model: EctCnnModel, graph_list: list):
    trainer = Trainer()

    trainer.test(model, dataloaders=DataLoader(graph_list))


def save(model: nn.Module):
    saved_model = model.state_dict()

    want_to_save: str = input("Want to save the model? (y/n)").lower()

    if want_to_save == "y":

        model_name = input("Want to save the model? Say the name: ")

        os.makedirs("saved_models", exist_ok=True)

        torch.save(saved_model, f"saved_models/{model_name}.pth")


def run():
    config = EctConfig()

    datasetGenerator = DatasetGenerator()

    train_graph_list, val_graph_list, test_graph_list = datasetGenerator.get_dataset(config)

    model = train(train_graph_list, config)

    model.eval()

    test(model, test_graph_list)

    save(model)

if __name__ == "__main__":
    run()