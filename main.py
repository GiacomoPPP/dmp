import os

import math

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import torch
from torch.nn import MSELoss
from torch.optim import SGD
import torch.nn as nn

from AssessmentDTO import AssessmentDTO
from DatasetGenerator import DatasetGenerator
from config import EctConfig

from model import EctCnnModel


def get_model_setup(graph_batch: list, config: EctConfig) -> tuple[EctCnnModel, DataLoader, SGD, MSELoss] :
    model = EctCnnModel(config)

    loader = DataLoader(graph_batch, config.n_minibatch, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    loss_function: MSELoss = nn.MSELoss()

    return model, loader, optimizer, loss_function


def clip_gradient(model, max_norm = 50):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm**2
    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
    return total_norm


def update_plot(assessment: AssessmentDTO):

    assessment.line1.set_xdata(assessment.epoch_list)
    assessment.line1.set_ydata(assessment.train_error_list)
    assessment.line2.set_xdata(assessment.epoch_list)
    assessment.line2.set_ydata(assessment.test_error_list)

    assessment.ax.relim()
    assessment.ax.autoscale_view()
    assessment.fig.canvas.draw()
    assessment.fig.canvas.flush_events()


def run_epoch_assessment(
        epoch: int,
        train_error: float,
        model: EctCnnModel,
        test_graph_list: list,
        config: EctConfig,
        assessment: AssessmentDTO
    ):

    assessment.epoch_list.append(epoch)

    assessment.train_error_list.append(train_error)

    test_error = test(model, test_graph_list, config)
    assessment.test_error_list.append(test_error)

    update_plot(assessment)

    print(f"Epoch {epoch} \n  Training loss {train_error:.2f} \n Test loss {test_error:.2f}")


def train(graph_batch: list, config: EctConfig, test_graph_list: list) -> nn.Module:

    model, loader, optimizer, loss_function = get_model_setup(graph_batch, config)

    assessment: AssessmentDTO = AssessmentDTO()

    for epoch in range(config.n_epochs):
        for mini_batch in loader:
            mini_batch.to(config.device)
            optimizer.zero_grad()
            predicted = model(mini_batch)
            loss = loss_function(predicted, mini_batch.y)
            loss.backward()
            clip_gradient(model)
            optimizer.step()
        if epoch > 0 and epoch % config.assessment_rate == 0:
            run_epoch_assessment(
                    epoch,
                    loss.item(),
                    model,
                    test_graph_list,
                    config,
                    assessment
                )

    return model


def test(model: nn.Module, graph_list: list, config: EctConfig):
    model.eval()
    with torch.no_grad():
        graph_batch: Batch = Batch.from_data_list(graph_list).to(config.device)
        predicted = model(graph_batch)
        loss = nn.MSELoss()(predicted, graph_batch.y).item()

        if math.isnan(loss):
            print("ouch")

    print(loss)

    return loss


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

    train_graph_list, test_graph_list = datasetGenerator.get_dataset(config)

    model = train(train_graph_list, config, test_graph_list)

    model.eval()

    save(model)


run()