from typing import Any, List, Tuple, Union, Iterable, Set, Dict

import numpy as np
import torch
from numpy import ndarray
from rdkit import Chem
from rdkit.Chem import Mol
from torch import LongTensor, Tensor, FloatTensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

BONDS: List[Union[object, str]] = [
    Chem.BondType.AROMATIC,
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    "other",
]  # Add more bonds

NUM_FEATURES: int = 6

HYBRIDIZATION: List[Union[object, str]] = [
    Chem.rdchem.HybridizationType.S,
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D2,
    Chem.rdchem.HybridizationType.OTHER,
    "other",
]

CHIRAL_TAGS: List[Union[object, str]] = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER,
    "other",
]

ATOMIC_NUMS: List[Union[int, str]] = list(range(1, 119)) + ["other"]
DEGREES: List[Union[int, str]] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, "other"]
FORMAL_CHARGES: List[Union[int, str]] = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, "other"]
NUM_HS: List[Union[int, str]] = [0, 1, 2, 3, 4, 5, 6, 7, 8, "other"]

IS_AROMATIC: List[bool] = [True, False]


def get_feature_index(feature: Any, features: List[Any]) -> int:
    if feature in features:
        return features.index(feature)

    return features[-1]


def get_atom_features_dims() -> List[int]:
    return [
        len(ATOMIC_NUMS),
        len(FORMAL_CHARGES),
        len(NUM_HS),
        len(IS_AROMATIC),
        len(HYBRIDIZATION),
        len(CHIRAL_TAGS),
    ]


def get_edge_index_with_attrs(molecule: Mol) -> Tuple[Tensor]:
    matrix: ndarray = Chem.GetAdjacencyMatrix(molecule)

    for bond in molecule.GetBonds():
        i: int = bond.GetBeginAtomIdx()
        j: int = bond.GetEndAtomIdx()
        bond_idx = (
            get_feature_index(bond.GetBondType(), BONDS) + 1
        )  # prevent zero bonds
        matrix[i, j] = bond_idx
        matrix[j, i] = bond_idx

    return dense_to_sparse(torch.tensor(matrix))


def get_atoms_feature_matrix(molecule: Mol) -> Tensor:
    feature_matrix: ndarray = np.zeros((molecule.GetNumAtoms(), NUM_FEATURES))

    for idx, atom in enumerate(molecule.GetAtoms()):
        atom_feature_vec: ndarray = np.array(
            [
                get_feature_index(atom.GetAtomicNum(), ATOMIC_NUMS),
                get_feature_index(atom.GetFormalCharge(), FORMAL_CHARGES),
                get_feature_index(atom.GetNumExplicitHs(), NUM_HS),
                get_feature_index(atom.GetIsAromatic(), IS_AROMATIC),
                get_feature_index(atom.GetHybridization(), HYBRIDIZATION),
                get_feature_index(atom.GetChiralTag(), CHIRAL_TAGS),
            ]
        )
        feature_matrix[idx] = atom_feature_vec
    return LongTensor(feature_matrix)


def mol_to_torch_data(molecule: Mol) -> Data:
    edge_index, edge_attr = get_edge_index_with_attrs(molecule)
    x = get_atoms_feature_matrix(molecule)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )


def add_label(graph: Data, label: Any) -> Data:
    graph.y = Tensor([[label]])
    return graph


def train_test_split(dataset: List, ratio: float) -> Tuple[List]:
    pointer: int = 1 - round(len(dataset) * ratio)
    return dataset[:pointer], dataset[pointer:]


class SmilesEncoder:

    def __init__(self, smiles: Iterable[str])-> None:
        self.smiles: Iterable[str] = smiles
        self._max_dim: int = self.max_dim
        self._alphabet: Set[str] = self.alphabet

    @property
    def max_dim(self)->int:
        return np.max([len(smile) for smile in self.smiles])

    @property
    def alphabet(self) -> Dict[str, int]:
        alphabet = set.union(*[set(smile)for smile in self.smiles])
        return {element:value for value, element in enumerate(alphabet)}

    def _encode_smile(self, smile:str)->ndarray:
        encode_mat = np.zeros((self._max_dim, len(self._alphabet)), dtype=np.int16)

        for idx, char in enumerate(smile):
            alphabet_pos = self._alphabet[char]
            encode_mat[idx, alphabet_pos] = 1

        return encode_mat.reshape(1, *encode_mat.shape)

    def transform(self)->ndarray:
        encoded_smiles: List[ndarray] = []

        for smile in self.smiles:
            encoded_smiles.append(self._encode_smile(smile))

        return np.array(encoded_smiles)


class SmilesDataSet(Dataset):
    def __init__(self, smiles: Iterable[str], labels:Iterable[Any])->None:
        self.smiles = smiles
        self.labels = labels
        self.smiles_encoder = SmilesEncoder(self.smiles)

    def __len__(self)->int:
        return len(self.smiles)

    def __getitem__(self, index: int) -> Tuple[ndarray, Any]:
        return FloatTensor(self.smiles_encoder._encode_smile(self.smiles[index])), FloatTensor([(self.labels[index])]),


class FingerprintDataset(Dataset):
    def __init__(self, fingerprints: ndarray) -> None:
        self.fingerprints = fingerprints
    
    def __len__(self)->int:
        return len(self.fingerprints)

    def __getitem__(self, index:int) -> ndarray:
        return FloatTensor(self.fingerprints[index])
