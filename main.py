import pickle

import os

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from pandas import DataFrame

import numpy as np

import math

import rdkit
import rdkit.Chem.Draw as Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Mol
from rdkit.Chem import AllChem

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import torch
from torch.nn import MSELoss
from torch.optim import SGD
import torch.nn as nn

from config import EctConfig

from model import EctCnnModel


n_samples = 2**9

n_minibatch = 32

n_epochs = 200

assessment_rate = 5

def random_split(df, test_size=0.2, seed=0):
    indices = np.arange(df.shape[0])

    _, _, _, _, train_index, test_index = train_test_split(
            df["smiles"],
            df["target"],
            indices,
            test_size=test_size,
            random_state=seed
        )

    return [[train_index, test_index]]


def prepare_dataset(n_samples):
    """
    Loads the features, filters nans and performs test-train splitting.
    """
    # Load processed dataset
    with open('data/features/ADRA1A/ADRA1A_2DAP.pkl', 'rb') as file:
        data: DataFrame = pickle.load(file)

    # Sample dataset
    data = data.sample(n=n_samples, random_state=2024).reset_index()
    data = data[~data["smiles"].isna()].reset_index()

    splits = random_split(data)

    # Apply splitting
    train_mask = splits[0][0]
    test_mask = splits[0][1]

    # Get data
    feature_cols = [col for col in data.columns if col not in [
        "smiles", "target", "index", "MurckoScaffold"]]
    X = data[feature_cols].values
    y = data["target"]

    return data, X, y, train_mask, test_mask


def extract_molecules_descriptors(data: DataFrame) -> tuple[list, list]:
    molecule_list = []
    descriptor_list = []

    for smile in data["smiles"]:
        molecule = rdkit.Chem.MolFromSmiles(smile)
        molecule_list.append(molecule)
        descriptor_list.append(Descriptors.MolLogP(molecule))

    return molecule_list, descriptor_list


def mol_to_graph(molecule: Mol, config: EctConfig, y: float) -> Data:
    molecule = rdkit.Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule)

    atom_features = []
    edge_index = []

    for atom in molecule.GetAtoms():
        positions = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
        atom_features.append([positions.x, positions.y, positions.z])

    for bond in molecule.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))

    return Data(
        x=torch.tensor(atom_features, dtype=torch.float, device = config.device),
        edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
        y = y
    )


def get_dataset(config: EctConfig) -> tuple[list, list]:
    data, X, y_list, train_mask, test_mask = prepare_dataset(n_samples)

    molecule_list, descriptor_list = extract_molecules_descriptors(data)

    graph_list: list = [mol_to_graph(molecule, config, y) for molecule, y in zip(molecule_list, y_list)]

    train_graph_list: list = [graph_list[index] for index in train_mask]

    test_graph_list: list = [graph_list[index] for index in test_mask]

    return train_graph_list, test_graph_list


def get_model_setup(graph_batch: list, config: EctConfig) -> tuple[EctCnnModel, DataLoader, SGD, MSELoss] :
    model = EctCnnModel(config)

    loader = DataLoader(graph_batch, n_minibatch, shuffle = True)

    optimizer = torch.optim.SGD(model.parameters())

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


def get_plot_setup():
    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], label='Test Error')
    line2, = ax.plot([], [], label='Eval Error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Error')
    ax.legend()
    plt.show()

    return fig, ax, line1, line2


def update_plot(
        epoch_list: list,
        train_error_list: list,
        evaluation_error_list: list,
        line1: Line2D,
        line2: Line2D,
        ax: Axes,
        fig: Figure
    ):

    line1.set_xdata(epoch_list)
    line1.set_ydata(train_error_list)
    line2.set_xdata(epoch_list)
    line2.set_ydata(evaluation_error_list)

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

def run_epoch_assessment(
        epoch: int,
        epoch_list: list,
        train_error: float,
        train_error_list,
        model: EctCnnModel,
        test_graph_list: list,
        evaluation_error_list: list,
        config: EctConfig,
        line1: Line2D,
        line2: Line2D,
        ax: Axes,
        fig: Figure
    ):
    epoch_list.append(epoch)

    train_error_list.append(train_error)

    Evaluation_loss = test(model, test_graph_list, config)
    evaluation_error_list.append(Evaluation_loss)

    update_plot(epoch_list, train_error_list, evaluation_error_list, line1, line2, ax, fig)

    print(f"Epoch {epoch} \n  Training loss {train_error:.2f} \n Evaluation loss {Evaluation_loss:.2f}")


def train(graph_batch: list, config: EctConfig, test_graph_list: list) -> nn.Module:

    model, loader, optimizer, loss_function = get_model_setup(graph_batch, test_graph_list)

    epoch_list, train_error_list, evaluation_error_list = [], [], []

    fig, ax, line1, line2 = get_plot_setup()

    for epoch in range(n_epochs):
        for mini_batch in loader:
            optimizer.zero_grad()
            mini_batch.to(config.device)
            predicted = model(mini_batch)
            loss = loss_function(predicted, mini_batch.y)
            loss.backward()
            clip_gradient(model)
            optimizer.step()
        if epoch > 0 and epoch % assessment_rate == 0:
            run_epoch_assessment(
                    epoch_list,
                    loss.item(),
                    train_error_list,
                    model,
                    test_graph_list,
                    evaluation_error_list,
                    config,
                    line1,
                    line2,
                    ax,
                    fig
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

    model_name = input("Want to save the model? Say the name: ")

    os.makedirs("saved_models", exist_ok=True)

    torch.save(saved_model, f"saved_models/{model_name}.pth")


def run():
    config = EctConfig()

    train_graph_list, test_graph_list = get_dataset(config)

    model = train(train_graph_list, config, test_graph_list)

    model.eval()

    save(model)


run()