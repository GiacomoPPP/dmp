import pickle

import os

import warnings

from sklearn.model_selection import train_test_split

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

from AssessmentDTO import AssessmentDTO
from config import EctConfig

from model import EctCnnModel


n_samples = 2**12

n_minibatch = 2**6

n_epochs = 400

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
    samples: int = min(len(data.index), n_samples)
    if samples < n_samples:
        warnings.warn(f"Number of samples required {n_samples} was higher than the available amount {samples}")

    data = data.sample(n=samples, random_state=2024).reset_index()
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

    graph_list: list = []

    for molecule, y in zip(molecule_list, y_list):
        try:
            molecule = mol_to_graph(molecule, config, y)
            graph_list.append(molecule)
        except ValueError:
            warnings.warn(f"Could not parse to graph molecule {molecule}")
            graph_list.append(None)

    train_graph_list: list = [graph_list[index] for index in train_mask]

    test_graph_list: list = [graph_list[index] for index in test_mask]

    return remove_none(train_graph_list), remove_none(test_graph_list)


def remove_none(graph_list: list) -> list:
    return [graph for graph in graph_list if graph is not None]


def get_model_setup(graph_batch: list, config: EctConfig) -> tuple[EctCnnModel, DataLoader, SGD, MSELoss] :
    model = EctCnnModel(config)

    loader = DataLoader(graph_batch, n_minibatch, shuffle = True)

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

    for epoch in range(n_epochs):
        for mini_batch in loader:
            mini_batch.to(config.device)
            optimizer.zero_grad()
            predicted = model(mini_batch)
            loss = loss_function(predicted, mini_batch.y)
            loss.backward()
            clip_gradient(model)
            optimizer.step()
        if epoch > 0 and epoch % assessment_rate == 0:
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

    train_graph_list, test_graph_list = get_dataset(config)

    model = train(train_graph_list, config, test_graph_list)

    model.eval()

    save(model)


run()