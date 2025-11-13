import os
from typing import Dict

from torch import nn
import torch
from DmpConfig import DmpConfig
from DmpDataset import DmpDataset
from TrainingRunner import TrainingRunner

config = DmpConfig()


class SingleTrainingRunner(TrainingRunner):
    def __call__(self):
        model, _ = self._train_and_assess_on_dataset(config.dataset)

        self.save(model)


    def save(self, model: nn.Module):
        if config.fast_run:
            return

        saved_model = model.state_dict()

        want_to_save: str = input("Want to save the model? (y/n)").lower()

        if want_to_save == "y":

            model_name = input("Want to save the model? Say the name: ")

            os.makedirs("saved_models", exist_ok=True)

            torch.save(saved_model, f"saved_models/{model_name}.pth")