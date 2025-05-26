import os

from torch_geometric.loader import DataLoader

import torch
from torch.nn import MSELoss
from torch.optim import SGD
import torch.nn as nn

from lightning import Trainer

from DatasetGenerator import DatasetGenerator
from Dmi1DModel import Dmi1DModel
from DmiConfig import DmiConfig

from DmiModel import DmiModel
from lightning.pytorch.loggers import TensorBoardLogger

class TrainingRunner:

    def __call__(self):
        config = DmiConfig()

        datasetGenerator = DatasetGenerator()

        train_graph_list, val_graph_list, test_graph_list = datasetGenerator.get_dataset(config)

        model = self.train(train_graph_list, val_graph_list, config)

        model.eval()

        self.test(model, test_graph_list)

        self.save(model, config)


    def get_model_setup(self, train_graph_list: list, val_graph_list: list, config: DmiConfig) -> tuple[DmiModel, DataLoader, SGD, MSELoss] :
        #model = Dmi1DModel(config)
        model = DmiModel(config)

        train_loader = DataLoader(train_graph_list, config.n_minibatch, shuffle = True)

        val_loader = DataLoader(val_graph_list, config.n_minibatch, shuffle = False)

        tensorboard_path: str = "tensorboard_sequential_smoothing"

        logger = TensorBoardLogger(tensorboard_path)

        trainer = Trainer(
                logger=logger,
                limit_train_batches = config.n_minibatch,
                max_epochs = config.n_epochs,
                log_every_n_steps = config.log_every_n_steps,
                fast_dev_run = config.fast_run,
            )

        return model, train_loader, val_loader, trainer


    def train(self, train_graph_list: list, val_graph_list:list, config: DmiConfig) -> nn.Module:

        model, train_loader, val_loader, trainer = self.get_model_setup(train_graph_list, val_graph_list, config)

        trainer.fit(model, train_loader, val_loader)

        return model


    def test(self, model: DmiModel, graph_list: list):
        trainer = Trainer()

        trainer.test(model, dataloaders=DataLoader(graph_list))


    def save(self, model: nn.Module, config: DmiConfig):
        if config.fast_run:
            return

        saved_model = model.state_dict()

        want_to_save: str = input("Want to save the model? (y/n)").lower()

        if want_to_save == "y":

            model_name = input("Want to save the model? Say the name: ")

            os.makedirs("saved_models", exist_ok=True)

            torch.save(saved_model, f"saved_models/{model_name}.pth")