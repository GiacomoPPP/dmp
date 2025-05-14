import pickle

import warnings

from pandas import DataFrame

import numpy as np

import rdkit
import rdkit.Chem.Draw as Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import Mol
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split

import torch

from torch_geometric.data import Data, Batch

from config import EctConfig



class DatasetGenerator:
    def random_split(self, df, test_size=0.15, val_size = 0.15, seed=0):
        indices = np.arange(df.shape[0])

        val_test_size: float = test_size + val_size

        _, val_test_smiles, _, val_test_target, train_index, val_test_index = train_test_split(
            df["smiles"],
            df["target"],
            indices,
            test_size = val_test_size,
            random_state=seed
        )

        test_relative_size: float = test_size / (test_size + val_size)

        _, _, _, _, val_index, test_index = train_test_split(
            val_test_smiles,
            val_test_target,
            val_test_index,
            test_size = test_relative_size,
            random_state=seed
        )

        return [[train_index, val_index, test_index]]


    def prepare_dataset(self, config: EctConfig):

        fast_run_n_samples: int = 16

        n_samples = fast_run_n_samples if config.fast_run else config.n_samples

        with open('data/features/ADRA1A/ADRA1A_2DAP.pkl', 'rb') as file:
            data: DataFrame = pickle.load(file)

        # Sample dataset
        samples: int = min(len(data.index), n_samples)
        if samples < n_samples:
            warnings.warn(f"Number of samples required {n_samples} was higher than the available amount {samples}")

        data = data.sample(n=samples, random_state=2024).reset_index()
        data = data[~data["smiles"].isna()].reset_index()

        splits = self.random_split(data)

        # Apply splitting
        train_mask = splits[0][0]
        val_mask = splits[0][1]
        test_mask = splits[0][2]

        # Get data
        feature_cols = [col for col in data.columns if col not in [
            "smiles", "target", "index", "MurckoScaffold"]]
        X = data[feature_cols].values
        y = data["target"]

        return data, X, y, train_mask, val_mask, test_mask


    def extract_molecules_descriptors(self, data: DataFrame) -> tuple[list, list]:
        molecule_list = []
        descriptor_list = []

        for smile in data["smiles"]:
            molecule = rdkit.Chem.MolFromSmiles(smile)
            molecule_list.append(molecule)
            descriptor_list.append(Descriptors.MolLogP(molecule))

        return molecule_list, descriptor_list


    def mol_to_graph(self, molecule: Mol, config: EctConfig, y: float) -> Data:
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


    def get_dataset(self, config: EctConfig) -> tuple[list, list]:
        data, X, y_list, train_mask, val_mask, test_mask = self.prepare_dataset(config)

        molecule_list, descriptor_list = self.extract_molecules_descriptors(data)

        graph_list: list = []

        for molecule, y in zip(molecule_list, y_list):
            try:
                molecule = self.mol_to_graph(molecule, config, y)
                graph_list.append(molecule)
            except ValueError:
                warnings.warn(f"Could not parse to graph molecule {molecule}")
                graph_list.append(None)

        train_graph_list: list = [graph_list[index] for index in train_mask]

        val_graph_list: list = [graph_list[index] for index in val_mask]

        test_graph_list: list = [graph_list[index] for index in test_mask]

        return self.remove_none(train_graph_list), self.remove_none(val_graph_list), self.remove_none(test_graph_list)

    def remove_none(self, graph_list: list) -> list:
        return [graph for graph in graph_list if graph is not None]
