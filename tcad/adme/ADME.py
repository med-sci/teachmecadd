import os
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

LABELS = "Molecular_weight Acceptors Donors LogP"
UNWANTED_PATH = "../tcad/adme/unwanted.txt"


class LipinskiCalc:
    def __init__(self, smiles):
        self.smiles = smiles
        self.props = self._get_lipinski_props()
        self.lipinski_table = self._props

    def _get_lipinski_props(self):
        mols = [Chem.MolFromSmiles(smile) for smile in self.smiles]
        props_df = pd.DataFrame(
            {
                "Smile": self.smiles,
                "Molecular_weight": [Descriptors.ExactMolWt(m) for m in mols],
                "Acceptors": [Lipinski.NumHAcceptors(m) for m in mols],
                "Donors": [Lipinski.NumHDonors(m) for m in mols],
                "LogP": [Crippen.MolLogP(m) for m in mols],
            }
        )
        return props_df

    def _check_props(self):
        fulfillment = []

        for i in range(self.props.shape[0]):
            s = sum(
                [
                    self.props["Molecular_weight"][i] <= 500,
                    self.props["Acceptors"][i] <= 10,
                    self.props["Donors"][i] <= 5,
                    self.props["LogP"][i] <= 5,
                ]
            )

            if s >= 4:
                fulfillment.append(True)
            else:
                fulfillment.append(False)

        return fulfillment

    @property
    def _props(self):
        lp = self._get_lipinski_props()
        f = self._check_props()
        lp["Fulfill"] = f

        return lp

    def _get_stats(self, df):
        means = []
        plus_stds = []
        minus_stds = []

        for column in LABELS.split():
            m = mean(df[column])
            std = stdev(df[column])

            if column == "Molecular_weight":
                m = m / 100
                std = stdev(df[column]) / 100
            elif column == "Acceptors":
                m = m / 2
                std = stdev(df[column]) / 2

            means.append(m)
            plus_stds.append(m + std)
            minus_stds.append(m - std)

        means = means + [means[0]]
        plus_stds = plus_stds + [plus_stds[0]]
        minus_stds = minus_stds + [minus_stds[0]]

        return means, plus_stds, minus_stds

    def visualize(self, df):
        line1, line2, line3 = self._get_stats(df)
        label_places = np.linspace(start=0, stop=2 * np.pi, num=len(line1))

        plt.figure(figsize=(6, 6))
        ax = plt.subplot(polar=True)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(180)

        plt.xticks(label_places, LABELS.split() + [""], fontsize=16)
        ax.fill(label_places, [5] * 5, "cornflowerblue", alpha=0.2)
        ax.plot(label_places, line1, "b", lw=3, ls="-")
        ax.plot(label_places, line2, "orange", lw=2, ls="--")
        ax.plot(label_places, line3, "orange", lw=2, ls="-.")
        labels = ("mean", "mean + std", "mean - std", "rule of five area")
        ax.legend(labels, loc=(1.1, 0.7), labelspacing=0.3, fontsize=16)

        plt.show()


class PAINS:
    def __init__(self, smiles):
        self._smiles = smiles
        self._pains = []

    def get_pains(self):
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog(params)
        matches = []
        pains = []

        for s in self._smiles:
            mol = Chem.MolFromSmiles(s)
            entry = catalog.GetFirstMatch(mol)

            if entry is not None:
                matches.append({"smiles": s, "pains": entry.GetDescription()})
                pains.append(s)
        self._pains = pains

        if len(matches) == 0:

            return "No pains found"
        else:

            return pd.DataFrame.from_dict(matches)

    @property
    def pains_free(self):
        self._smiles = [s for s in self._smiles if s not in self._pains]

        return self._smiles


class UnwantedSearcher:
    def __init__(self, smiles):
        self.unwanted_subs = self._unwanted
        self._smiles = smiles

    @property
    def _unwanted(self):
        unwanted = {}

        with open(UNWANTED_PATH, "r") as f:

            for line in f.readlines():
                unwanted[line.split()[0]] = line.split()[1]
        return unwanted

    def get_unwanted(self):
        matches = []
        clean = []

        for smile in self._smiles:
            mol = Chem.MolFromSmiles(smile)
            clear = True

            for name, smarts in self.unwanted_subs.items():

                if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                    matches.append({"smiles": smile, "name": name, "smarts": smarts})
                    clear = False
            if clear:
                clean.append(smile)

        return matches, clean

    def visualize(self, u_subs):
        matches = []
        mols = []
        names = []

        for i in u_subs:
            mol = Chem.MolFromSmiles(i["smiles"])
            matches.append(mol.GetSubstructMatch(Chem.MolFromSmarts(i["smarts"])))
            names.append(i["name"])
            mols.append(mol)
        return Draw.MolsToGridImage(
            mols, highlightAtomLists=matches, legends=names, molsPerRow=5
        )
