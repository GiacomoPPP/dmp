import os

from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn

from lightning import Trainer

from DatasetGenerator import DatasetGenerator
from DmpConfig import DmpConfig

from DmpDataset import DmpDataset
from DmpModel import DmpModel
from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch.callbacks.early_stopping import EarlyStopping

config = DmpConfig()
class TrainingRunner:

    def __call__(self):

        datasetGenerator = DatasetGenerator()

        for dataset in DmpDataset:
            self._train_and_assess_on_dataset(datasetGenerator, dataset)


    def _train_and_assess_on_dataset(self, datasetGenerator: DatasetGenerator, dataset: DmpDataset):
            train_graph_list, val_graph_list, test_graph_list, geometric_scale = datasetGenerator.get_dataset(dataset, config.include_hydrogens_in_training)

            model = self._train(train_graph_list, val_graph_list, geometric_scale)

            model.eval()

            self.test(model, test_graph_list)

            self.save(model)


    def _train(self, train_graph_list: list, val_graph_list:list, geometric_scale: float) -> nn.Module:

        model, train_loader, val_loader, trainer = self._get_model_setup(train_graph_list, val_graph_list)

        model.geometric_scale.copy_(geometric_scale)

        trainer.fit(model, train_loader, val_loader)

        return model


    def _get_model_setup(self, train_graph_list: list, val_graph_list: list) -> tuple[DmpModel, DataLoader, DataLoader, Trainer]:

        model = DmpModel()

        train_loader = DataLoader(train_graph_list, config.n_minibatch, shuffle = True)

        val_loader = DataLoader(val_graph_list, config.n_minibatch, shuffle = False)

        tensorboard_path: str = "tensorboard_v7"

        logger = TensorBoardLogger(tensorboard_path)

        trainer = Trainer(
                logger=logger,
                limit_train_batches = 0.2,
                max_epochs = config.n_epochs,
                log_every_n_steps = config.log_every_n_steps,
                fast_dev_run = config.fast_run,
                check_val_every_n_epoch = config.check_val_every_n_epoch,
                callbacks = [self._get_early_stop_callback()]
            )

        return model, train_loader, val_loader, trainer


    def _get_early_stop_callback(self) -> EarlyStopping:
        return EarlyStopping(monitor="val_loss", mode="min", patience=7)


    def test(self, model: DmpModel, graph_list: list):
        trainer = Trainer()

        test_loss = trainer.test(model, dataloaders=DataLoader(graph_list))[0].get('test_loss')

        model.register_buffer("test_loss", torch.tensor(test_loss))


    def save(self, model: nn.Module):
        if config.fast_run:
            return

        saved_model = model.state_dict()

        want_to_save: str = input("Want to save the model? (y/n)").lower()

        if want_to_save == "y":

            model_name = input("Want to save the model? Say the name: ")

            os.makedirs("saved_models", exist_ok=True)

            torch.save(saved_model, f"saved_models/{model_name}.pth")