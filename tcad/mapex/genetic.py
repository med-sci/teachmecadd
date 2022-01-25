# TODO keboard interrupt

import random
from datetime import datetime
from statistics import mean, stdev

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, SDWriter, rdMolDescriptors

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def log(message):
    print(f"{datetime.utcnow()} | INFO | {message}")


class GA:
    def __init__(
        self,
        smiles,
        n_conformers,
        n_inds,
        mutation_chance,
        generations,
        crippen=False,
        verbose=True,
    ):
        self._smiles = smiles
        self._nconfs = n_conformers
        self._n_inds = n_inds
        self._mut_chance = mutation_chance
        self._num_gens = generations
        self._useCrippen = crippen
        self._cache = []
        self.population = []
        self.best_chromosome = None
        self.verbose = verbose

    def _set_up(self):
        self._mols = self._create_confs()
        self._ref = self._mols[0]
        self._prbs = self._mols[1:]
        self._MMFFs = self._MMFF_props
        self._Crippens = self._crippen_contribs
        self._population = self.__population
        self.RMSDs = []

    def _create_confs(self):
        mols = []
        log(
            "Generating conformations..",
        )

        for smile in self._smiles:
            mol = Chem.MolFromSmiles(smile)
            mol_hs = Chem.AddHs(mol)
            _ = AllChem.EmbedMultipleConfs(mol_hs, self._nconfs)
            mols.append(Chem.RemoveHs(mol_hs))

        log("Done..")
        return mols

    class _Chromosome:
        def __init__(self, n_confs, n_genes):
            self._n_confs = n_confs
            self._n_genes = n_genes
            self.score = None
            self.chromosome = self._chromosome

        def __repr__(self):
            return "chromosome"

        @property
        def _chromosome(self):
            chromosome = [
                random.randint(0, self._n_confs - 1) for _ in range(self._n_genes)
            ]
            return chromosome

    def _cache_population(self, population):
        for chromosome in population:

            if chromosome.chromosome not in self._cache:
                self._cache.append(chromosome.chromosome)
        return self._cache

    @property
    def _MMFF_props(self):
        return [AllChem.MMFFGetMoleculeProperties(m) for m in self._mols]

    @property
    def _crippen_contribs(self):
        return [rdMolDescriptors._CalcCrippenContribs(m) for m in self._mols]

    @property
    def __population(self):
        return [
            self._Chromosome(self._mols[0].GetNumConformers(), len(self._mols))
            for _ in range(self._n_inds)
        ]

    def __fitness(self, chromosome):
        if chromosome.chromosome in self._cache:
            return chromosome
        else:
            ref = self._ref
            prbs = self._prbs
            scores = []

            for i in range(len(prbs)):

                if self._useCrippen:
                    o3a = AllChem.GetCrippenO3A(
                        prbs[i],
                        ref,
                        self._crippen_contribs[i + 1],
                        self._crippen_contribs[0],
                        prbCid=chromosome.chromosome[i + 1],
                        refCid=chromosome.chromosome[0],
                    )
                else:
                    o3a = AllChem.GetO3A(
                        prbs[i],
                        ref,
                        self._MMFFs[i + 1],
                        self._MMFFs[0],
                        prbCid=chromosome.chromosome[i + 1],
                        refCid=chromosome.chromosome[0],
                    )
                rmsd = o3a.Align()
                self.RMSDs.append(rmsd)
                matches = o3a.Matches()
                score = len(matches) / ref.GetNumAtoms()
                scores.append(score)
            total = mean(scores)
            chromosome.score = total
            return chromosome

    def __order_population(self, population):
        return sorted(population, key=lambda chromosome: chromosome.score, reverse=True)

    def __crossover(self, population):
        splt = random.randint(1, len(population[0].chromosome) - 1)

        if len(population) % 2 == 0:
            len_pop = len(population) - 1
        else:
            len_pop = len(population)

        for i in range(1, len_pop, 2):
            children1 = (
                population[i].chromosome[0:splt] + population[i + 1].chromosome[splt:]
            )
            children2 = (
                population[i + 1].chromosome[0:splt] + population[i].chromosome[splt:]
            )
            population[i].chromosome = children1
            population[i + 1].chromosome = children2
        return population

    def __mutation(self, chromosome, chance):
        assert chance > 0 and chance <= 1
        distribution = [chance, 1 - chance]
        random_number = random.choices([1, 0], distribution)

        if random_number[0] == 1:
            gene = random.randint(0, len(chromosome.chromosome) - 1)
            chromosome.chromosome[gene] = random.randint(0, chromosome._n_confs - 1)
        return chromosome

    def run(self, logg_iter):
        self._set_up()

        population = self._population

        for gen in range(self._num_gens):
            population = [self.__fitness(chromosome) for chromosome in population]
            population = self.__order_population(population)
            self._cache_population(population)

            if self.verbose and gen % logg_iter == 0:
                log(
                    f"Best chromosome score is: {round(population[0].score, 3)} | run {gen+1}"
                )

            population = self.__crossover(population)
            population = [population[0]] + [
                self.__mutation(chromosome, self._mut_chance)
                for chromosome in population[1:]
            ]
        population = [self.__fitness(chromosome) for chromosome in population]
        population = self.__order_population(population)

        if self.verbose:
            log(
                f"Best chromosome score is: {round(population[0].score, 3)} | run {gen+1}"
            )

        self.population = population
        self.best_chromosome = population[0].chromosome

    def get_molecules(self):
        ref = self._ref
        prbs = self._prbs
        chromosome = self.best_chromosome

        for i, mol in enumerate(prbs):
            o3a = AllChem.GetO3A(
                mol,
                ref,
                self._MMFFs[i + 1],
                self._MMFFs[0],
                prbCid=chromosome[i + 1],
                refCid=chromosome[0],
            )
            o3a.Align()
        mols = [ref] + prbs
        return mols

    def write(self, molecules, chromosome, path):
        with Chem.SDWriter(path) as writer:

            for molecule, conformation in zip(molecules, chromosome):
                writer.write(molecule, confId=conformation)
