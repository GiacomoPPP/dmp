from abc import ABC
import os
from typing import Dict

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

from ModelAssessment import ModelAssessment

config = DmpConfig()
class TrainingRunner(ABC):

    def __init__(self):
        self.modelAssessment = ModelAssessment()

        self.datasetGenerator = DatasetGenerator()


    def _train_and_assess_on_dataset(self, dataset: DmpDataset) -> tuple[DmpModel, float]:
        train_graph_list, val_graph_list, test_graph_list, geometric_scale = \
            self.datasetGenerator.get_dataset(dataset, config.include_hydrogens_in_training)

        complete_dataset = train_graph_list + val_graph_list + test_graph_list

        root_mean_square = self.modelAssessment.compute_root_mean_square(complete_dataset)

        model = self._train(train_graph_list, val_graph_list, geometric_scale, root_mean_square)

        model.eval()

        relative_test_loss: float = self.modelAssessment.test(model, test_graph_list)

        return model, relative_test_loss


    def _train(self, train_graph_list: list, val_graph_list:list, geometric_scale: float, root_mean_square: float) -> nn.Module:

        model, train_loader, val_loader, trainer = self._get_model_setup(train_graph_list, val_graph_list, root_mean_square)

        model.geometric_scale.copy_(geometric_scale)

        trainer.fit(model, train_loader, val_loader)

        return model


    def _get_model_setup(self, train_graph_list: list, val_graph_list: list, root_mean_square: float) -> tuple[DmpModel, DataLoader, DataLoader, Trainer]:

        model = DmpModel(root_mean_square)

        train_loader = DataLoader(train_graph_list, config.n_minibatch, shuffle = True)

        val_loader = DataLoader(val_graph_list, config.n_minibatch, shuffle = False)

        tensorboard_path: str = f"tensorboard/{config.model_name}"

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
        return EarlyStopping(monitor="validation_pmse", mode="min", patience=7)
