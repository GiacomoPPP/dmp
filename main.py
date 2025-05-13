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
from lightning.pytorch.loggers import TensorBoardLogger

def get_model_setup(train_graph_list: list, val_graph_list: list, config: EctConfig) -> tuple[EctCnnModel, DataLoader, SGD, MSELoss] :
    model = EctCnnModel(config)

    train_loader = DataLoader(train_graph_list, config.n_minibatch, shuffle = True)

    val_loader = DataLoader(val_graph_list, config.n_minibatch, shuffle = False)

    tensorboard_path: str = "tensorboard"

    logger = TensorBoardLogger(tensorboard_path)

    trainer = Trainer(
            logger=logger,
            limit_train_batches = config.n_minibatch,
            max_epochs = config.n_epochs,
            log_every_n_steps = config.log_every_n_steps,
            fast_dev_run = config.fast_run,
        )

    return model, train_loader, val_loader, trainer


def train(train_graph_list: list, val_graph_list:list, config: EctConfig) -> nn.Module:

    model, train_loader, val_loader, trainer = get_model_setup(train_graph_list, val_graph_list, config)

    trainer.fit(model, train_loader, val_loader)

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

    model = train(train_graph_list, val_graph_list, config)

    model.eval()

    test(model, test_graph_list)

if __name__ == "__main__":
    run()