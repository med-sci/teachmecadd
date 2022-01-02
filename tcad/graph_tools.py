import rdkit 
from rdkit import Chem
import torch
import numpy as np
from rdkit.Chem.rdchem import Mol
from torch import Tensor
from typing import List, Tuple, Any
from numpy import ndarray
from torch_geometric.utils import dense_to_sparse 
from torch_geometric.data import Data


BONDS = {
    Chem.BondType.AROMATIC:1,
    Chem.BondType.SINGLE:2,
    Chem.BondType.DOUBLE:3,
    Chem.BondType.TRIPLE:4,
} # Add more bonds 

NUM_FEATURES = 6

HYBRIDIZATION = {
    Chem.rdchem.HybridizationType.S:0,
    Chem.rdchem.HybridizationType.SP:1,
    Chem.rdchem.HybridizationType.SP2:2,
    Chem.rdchem.HybridizationType.SP3:3,
    Chem.rdchem.HybridizationType.SP3D2:4,
    Chem.rdchem.HybridizationType.OTHER:5,
 }

CHIRAL_TAGS = {
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED:0,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW:1,
    Chem.rdchem.CHI_TETRAHEDRAL_CCW:2,
    Chem.rdchem.ChiralType.CHI_OTHER:3,
}

def get_edge_index_with_attrs(molecule: Mol) -> Tuple[Tensor]:
    matrix: ndarray =  Chem.GetAdjacencyMatrix(molecule)
    
    for bond in molecule.GetBonds():
        i: int= bond.GetBeginAtomIdx()
        j: int= bond.GetEndAtomIdx()
        matrix[i,j] = BONDS[bond.GetBondType()]
        matrix[j,i] = BONDS[bond.GetBondType()]

    return dense_to_sparse(torch.tensor(matrix))

def get_atoms_feature_matrix(molecule: Mol) -> ndarray:
    feature_matrix: ndarray = np.zeros((molecule.GetNumAtoms(), NUM_FEATURES))
    
    for idx, atom in enumerate(molecule.GetAtoms()):
        atom_feature_vec: ndarray = np.array([
            atom.GetAtomicNum(),
            atom.GetFormalCharge(),
            atom.GetNumExplicitHs(),
            int(atom.GetIsAromatic()),
            HYBRIDIZATION[atom.GetHybridization()],
            CHIRAL_TAGS[atom.GetChiralTag()],
            ])
        feature_matrix[idx] = atom_feature_vec
    return feature_matrix

def mol_to_torch_data(molecule: Mol)->Data:
    edge_index, edge_attr = get_edge_index_with_attrs(molecule) 
    return Data(
        x=get_atoms_feature_matrix(molecule), 
        edge_index=edge_index, edge_attr=edge_attr,
        )

def add_label(graph: Data, label: Any)->Data:
    graph.y = label
    return graph

def train_test_split(dataset: List, ratio: float)-> Tuple[List]:
    pointer = 1-round(len(dataset)*ratio)
    return dataset[:pointer], dataset[pointer:]