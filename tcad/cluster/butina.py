from typing import List

import pandas as pd
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import DataStructs, Draw, rdFingerprintGenerator
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Cluster import Butina


class ButinaClustering:

    RDKIT_FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)

    def __init__(
        self, smiles: List[str], chembl_indices: List[str], cut_off: float = 0.2
    ):
        self.smiles: List[str] = smiles
        self.chembl_indices: List[str] = chembl_indices
        self.cut_off: float = cut_off
        self._generate_molecules()
        self._generate_fingerprints()

    def _generate_molecules(self):
        self.molecules: List[Mol] = [Chem.MolFromSmiles(smile) for smile in self.smiles]

    def _generate_fingerprints(self):
        self.fingerprints: List[object] = []

        for molecule in self.molecules:
            self.fingerprints.append(
                self.RDKIT_FINGERPRINT_GENERATOR.GetFingerprint(molecule)
            )

    def _get_tanimoto_distance_matrix(self):
        self.dissimilarity_matrix: List[List] = []

        for i in range(1, len(self.fingerprints)):
            similarities = DataStructs.BulkTanimotoSimilarity(
                self.fingerprints[i], self.fingerprints[:i]
            )
            self.dissimilarity_matrix.extend([1 - x for x in similarities])

    def cluster(self):
        self._get_tanimoto_distance_matrix()
        clusters = Butina.ClusterData(
            self.dissimilarity_matrix,
            len(self.fingerprints),
            self.cut_off,
            isDistData=True,
        )
        self.clusters = sorted(clusters, key=len, reverse=True)

    def get_molecules(self, cluster_idx):
        cluster = self.clusters[cluster_idx]
        return [self.molecules[i] for i in cluster]

    def plot_clusters(self):
        _, ax = plt.subplots(figsize=(15, 4))
        ax.set_title("Distribution of clusters")
        ax.set_xlabel("Cluster index")
        ax.set_ylabel("Number of molecules")
        ax.bar(
            range(0, len(self.clusters)),
            [len(cluster) for cluster in self.clusters],
            lw=5,
        )

        plt.show()

    def plot_moleculs_of_cluster(self, cluster, num_mols):
        num_mols = (
            len(self.clusters[cluster])
            if num_mols > len(self.clusters[cluster])
            else num_mols
        )

        return Draw.MolsToGridImage(
            [self.molecules[self.clusters[cluster][i]] for i in range(num_mols)],
            legends=[
                self.chembl_indices[self.clusters[cluster][i]] for i in range(num_mols)
            ],
            molsPerRow=5,
        )

    def plot_centroids(self, n_clusters):
        return Draw.MolsToGridImage(
            [self.molecules[self.clusters[i][0]] for i in range(n_clusters)],
            legends=[
                self.chembl_indices[self.clusters[i][0]] for i in range(n_clusters)
            ],
            molsPerRow=5,
        )
