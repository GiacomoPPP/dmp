import os
from pathlib import Path
import pickle
import warnings

from numpy import ndarray
from pandas import DataFrame
import torch
from DmpConfig import DmpConfig
from DmpDataset import DmpDataset

from torch_geometric.data import Data as Graph

import rdkit
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import Mol
from rdkit.Chem import AllChem

config = DmpConfig()

class DatasetParser:

    def __init__(self):
        self.stored_dataset_path = "data/parsed"


    def __call__(self):
        self._parse_all_datasets()


    def load_parsed_dataset(self, dataset: DmpDataset) -> list[Graph]:

        stored_graph_list_file_path: str = self._get_parsed_dataset_path(dataset)

        with open(stored_graph_list_file_path, 'rb') as f:
            return pickle.load(f)


    def _get_parsed_dataset_path(self, dataset: DmpDataset):
        return f"{self.stored_dataset_path}/{dataset}.pkl"


    def _parse_all_datasets(self):
        for dataset in DmpDataset:
            data = self._load_from_file(dataset)

            parsed_data_list: list[Graph] = self._parse_to_graph_list(data, config.add_hydrogens)

            self._store_dataset(parsed_data_list, dataset)


    def _store_dataset(self, graph_list: list[Graph], dataset: DmpDataset) -> None:
        parsed_dataset_path = self._get_parsed_dataset_path(dataset)

        os.makedirs(os.path.dirname(parsed_dataset_path), exist_ok=True)

        print("Saving to:", parsed_dataset_path)
        with open(parsed_dataset_path, 'wb') as f:
            pickle.dump(graph_list, f)


    def _parse_to_graph_list(self, data: DataFrame, add_hydrogens: bool) -> list[Graph]:

        target_list: ndarray = data["target"]

        molecule_list = self._extract_molecules_descriptors(data)

        graph_list: list[Graph] = []

        RDLogger.DisableLog('rdApp.*')

        for molecule, target in zip(molecule_list, target_list):
            try:
                molecule = self._mol_to_graph(molecule, target, add_hydrogens)
                graph_list.append(molecule)
            except ValueError:
                warnings.warn(f"Could not parse to graph molecule with SMILE {Chem.MolToSmiles(molecule)}")

        return graph_list


    def _load_from_file(self, dataset: DmpDataset) -> DataFrame:

        with open(self._get_raw_dataset_path(dataset), 'rb') as file:
            data: DataFrame = pickle.load(file)

        data = data[~data["smiles"].isna()].reset_index()

        return data


    def _get_raw_dataset_path(self, dataset: DmpDataset) -> str:
        base = base = Path(__file__).resolve().parent
        return f"{base}/data/features/{dataset.value}/{dataset.value}_2DAP.pkl"


    def _extract_molecules_descriptors(self, data: DataFrame) -> list[Mol]:
        molecule_list = []

        for smile in data["smiles"]:
            molecule = rdkit.Chem.MolFromSmiles(smile)
            molecule_list.append(molecule)

        return molecule_list


    def _mol_to_graph(self, molecule: Mol, y: float, add_hydrogens: bool) -> Graph:

        if(add_hydrogens):
            molecule = rdkit.Chem.AddHs(molecule)

        AllChem.EmbedMolecule(molecule)

        atom_coordinates = []
        edge_index = []

        for atom in molecule.GetAtoms():
            positions = molecule.GetConformer().GetAtomPosition(atom.GetIdx())
            atom_coordinates.append([positions.x, positions.y, positions.z])

        for bond in molecule.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index.append((i, j))
            edge_index.append((j, i))

        return Graph(
                x=torch.tensor(atom_coordinates, dtype=torch.float, device = config.device),
                edge_index=torch.tensor(edge_index, dtype=torch.long).t().contiguous(),
                y = y
            )