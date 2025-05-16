import pickle

import warnings

from pandas import DataFrame
from pandas import Series

from numpy import ndarray
import rdkit
from rdkit.Chem import Descriptors
from rdkit.Chem import Mol
from rdkit.Chem import AllChem

import torch

from torch_geometric.data import Data

from DatasetSplitter import DatasetSplitter
from DmiConfig import DmiConfig


class DatasetGenerator:

    def _load_from_file(self, config: DmiConfig) -> DataFrame:
        fast_run_n_samples: int = 128

        n_samples = fast_run_n_samples if config.fast_run else config.n_samples

        with open('data/features/ADRA1A/ADRA1A_2DAP.pkl', 'rb') as file:
            data: DataFrame = pickle.load(file)

        samples: int = min(len(data.index), n_samples)

        if samples < n_samples:
            warnings.warn(f"Number of samples required {n_samples} was higher than the available amount {samples}")

        data = data.sample(n=samples, random_state=2025).reset_index()
        data = data[~data["smiles"].isna()].reset_index()

        return data


    def _prepare_dataset(self, config: DmiConfig) -> tuple[DataFrame, Series, ndarray, ndarray, ndarray]:

        data = self._load_from_file(config)

        datasetSplitter = DatasetSplitter()

        target_list: ndarray = data["target"]

        train_size = 0.7
        test_size = 0.15
        val_size = 0.15

        splits = datasetSplitter(target_list.to_numpy(), train_size, test_size, val_size)

        train_mask = splits[0]
        val_mask = splits[1]
        test_mask = splits[2]

        return data, target_list, train_mask, val_mask, test_mask


    def _extract_molecules_descriptors(self, data: DataFrame) -> tuple[list, list]:
        molecule_list = []
        descriptor_list = []

        for smile in data["smiles"]:
            molecule = rdkit.Chem.MolFromSmiles(smile)
            molecule_list.append(molecule)
            descriptor_list.append(Descriptors.MolLogP(molecule))

        return molecule_list, descriptor_list


    def _mol_to_graph(self, molecule: Mol, config: DmiConfig, y: float) -> Data:
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


    def _remove_none(self, graph_list: list) -> list:
        return [graph for graph in graph_list if graph is not None]


    def get_dataset(self, config: DmiConfig) -> tuple[list, list, list]:
        data, target_list, train_mask, val_mask, test_mask = self._prepare_dataset(config)

        molecule_list, descriptor_list = self._extract_molecules_descriptors(data)

        graph_list: list = []

        for molecule, target in zip(molecule_list, target_list):
            try:
                molecule = self._mol_to_graph(molecule, config, target)
                graph_list.append(molecule)
            except ValueError:
                warnings.warn(f"Could not parse to graph molecule {molecule}")
                graph_list.append(None)

        train_graph_list: list = [graph_list[index] for index in train_mask]

        val_graph_list: list = [graph_list[index] for index in val_mask]

        test_graph_list: list = [graph_list[index] for index in test_mask]

        return self._remove_none(train_graph_list), self._remove_none(val_graph_list), self._remove_none(test_graph_list)

