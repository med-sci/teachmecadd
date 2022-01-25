import math
import os

import numpy as np
import py3Dmol
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from sklearn import cluster


class Pharmacophore:
    def __init__(self, mol, gene):
        self._fdef = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        self._mol = mol
        self._gene = gene
        self._factory = ChemicalFeatures.BuildFeatureFactory(self._fdef)
        self.acceptors, self.donors, self.hydrophobics = self._features

    @property
    def _features(self):
        mol = self._mol
        acceptors = [
            list(f.GetPos(confId=self._gene))
            for f in self._factory.GetFeaturesForMol(
                mol, includeOnly="Acceptor", confId=self._gene
            )
        ]
        donors = [
            list(f.GetPos(confId=self._gene))
            for f in self._factory.GetFeaturesForMol(
                mol, includeOnly="Donor", confId=self._gene
            )
        ]
        hydrophobics = [
            list(f.GetPos(confId=self._gene))
            for f in self._factory.GetFeaturesForMol(
                mol, includeOnly="Hydrophobe", confId=self._gene
            )
        ]
        return acceptors, donors, hydrophobics


class PharmComplex:
    def __init__(self, molecules, chromosome):
        self.molecules = molecules
        self.chromosome = chromosome
        self._p_list = [
            Pharmacophore(m, g) for m, g in zip(self.molecules, self.chromosome)
        ]
        self._feat_dict = self._features
        self._coords = {}

    def get_coords(self):
        return self._coords

    @property
    def _features(self):
        accs = []
        dons = []
        hyds = []

        for pharmacophore in self._p_list:
            accs.extend(pharmacophore.acceptors)
            dons.extend(pharmacophore.donors)
            hyds.extend(pharmacophore.hydrophobics)
        feat_dict = {"Acceptors": accs, "Donors": dons, "Hydrophobics": hyds}
        return feat_dict

    @property
    def _num_mols(self):
        return math.ceil(len(self._p_list) * 0.75)

    def _cluster_feats(self, feats, eps=1, min_samples=4):
        dbscan = cluster.DBSCAN(eps, min_samples=min_samples)
        dbscan.fit(feats)
        lbls = dbscan.labels_
        n_clus = len(np.unique(dbscan.labels_)) - 1
        clus_dict = {}

        for k in range(n_clus):
            clus_dict[k] = []

        for i in range(len(feats)):

            for j in range(n_clus):

                if lbls[i] == j:
                    clus_dict[j].append(feats[i])
        return clus_dict

    def _get_centers(self, clus_dict):
        centr_list = []

        for k in clus_dict.keys():
            x, y, z = (0, 0, 0)

            for i in clus_dict[k]:
                x = x + i[0]
                y = y + i[1]
                z = z + i[2]
            den = len(clus_dict[k])
            centr_list.append(
                [round((x / den), 4), round((y / den), 4), round((z / den), 4)]
            )
        return centr_list

    def create(self, distance=1, n_features=None):

        if n_features:
            num_mols = n_features
        else:
            num_mols = self._num_mols

        feat_dict = self._feat_dict
        coords = {}

        for k in feat_dict.keys():
            clus_dict = self._cluster_feats(feat_dict[k], distance, num_mols)
            centr_list = self._get_centers(clus_dict)
            coords[k] = centr_list

        self._coords = coords

    def show(self, mol_blocks, coords):
        xyzview = py3Dmol.view(width=1000, height=400)

        for mol_block in mol_blocks:
            xyzview.addModel(mol_block)

        xyzview.setStyle({"stick": {}})
        xyzview.setBackgroundColor("0xeeeeee")

        for key, value in coords.items():
            radius = 1.25
            color = "purple"
            label = "Acceptor"

            if key == "Donors":
                radius = 1
                color = "blue"
                label = "Donor"

            if key == "Hydrophobics":
                radius = 0.75
                color = "gray"
                label = "Hydrophobic"

            for coord_set in value:
                center = {"x": coord_set[0], "y": coord_set[1], "z": coord_set[2]}
                xyzview.addSphere(
                    {
                        "radius": radius,
                        "color": color,
                        "center": center,
                        "wireframe": True,
                    }
                )
                xyzview.addLabel(
                    label,
                    {
                        "position": center,
                        "backgroundColor": color,
                        "backgroundOpacity": 0.8,
                    },
                )

        return xyzview
