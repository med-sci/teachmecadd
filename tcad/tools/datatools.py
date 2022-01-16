from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import rdkit
import requests
from networkx import Graph
from numpy import ndarray
from pandas import DataFrame
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from requests.models import HTTPError, Response


class DataLoader:
    ACTIVITY_URL: str = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    MAX_RECORDS_ON_PAGE: int = 1000
    LAST_PAGE_INDEX: int = 1
    OK_RESPONSE_STATUS_CODE: int = 200

    def __init__(self, target: str) -> None:
        self.target: str = target
        self.activities: List[Dict] = []
        self._get_num_records()
        self._create_pagination()

    def _get_num_records(self) -> None:
        url: str = (
            self.ACTIVITY_URL + f"?limit=5&offset=0&target_chembl_id={self.target}"
        )
        response: Response = requests.get(url)
        self.num_records: int = response.json()["page_meta"]["total_count"]

    def _create_pagination(self) -> None:
        self.num_pages: int = 0

        if self.num_records <= self.MAX_RECORDS_ON_PAGE:
            self.num_full_pages: int = 1
            self.last_page_count: int = self.num_records
        else:
            self.num_full_pages: int = (
                self.num_records // self.MAX_RECORDS_ON_PAGE
            ) + self.LAST_PAGE_INDEX
            self.last_page_count: int = self.num_records % self.MAX_RECORDS_ON_PAGE

    def load_data(self) -> None:
        offset: int = 0

        if self.num_full_pages != self.LAST_PAGE_INDEX:

            for _ in range(self.num_full_pages):
                url: str = (
                    self.ACTIVITY_URL
                    + f"?limit={self.MAX_RECORDS_ON_PAGE}&offset={offset}&target_chembl_id={self.target}"
                )
                response: Response = requests.get(url)

                if response.status_code == self.OK_RESPONSE_STATUS_CODE:
                    self.activities += response.json()["activities"]
                    offset += self.MAX_RECORDS_ON_PAGE
                else:
                    raise HTTPError(f"Got response code: {response.status_code}")

        url: str = (
            self.ACTIVITY_URL
            + f"?limit={self.last_page_count}&offset={offset}&target_chembl_id={self.target}"
        )
        response: Response = requests.get(url)

        if response.status_code == self.OK_RESPONSE_STATUS_CODE:
            self.activities += response.json()["activities"]
        else:
            raise HTTPError(f"Got response code: {response.status_code}")

    def get_data(self) -> DataFrame:
        return pd.DataFrame.from_dict(self.activities)


def preprocess(data: DataFrame, *args: Tuple[str]) -> DataFrame:
    if not args:
        raise ValueError("Must provide columns arguments")

    data: DataFrame = data[list(args)]
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    return data


def to_log_p(array: Sequence[float]) -> Sequence[float]:
    return -np.log10(np.array(array) * 10 ** -9)


def check_smile(smile):
    mol = Chem.MolFromSmiles(smile)
    
    if mol == None:
        return "Not Valid"
    
    return "Valid"