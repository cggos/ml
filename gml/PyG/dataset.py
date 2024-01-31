#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Project : dl_with_pytorch 
@File    : dataset.py
@Site    : 
@Author  : Gavin Gao
@Date    : 11/26/22 9:26 PM 
"""
import torch
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data
from rdkit import Chem  # pip install rdkit
import pandas as pd


class COVID(InMemoryDataset):
    url = 'https://github.com/yangkevin2/coronavirus_data/raw/master/data/mpro_xchem.csv'

    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(COVID, self).__init__(root, transform, pre_transform, pre_filter)
        # Load processed data
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['mpro_xchem.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for smiles, label in df.itertuples(False, None):
            mol = Chem.MolFromSmiles(smiles)  # Read the molecule info
            adj = Chem.GetAdjacencyMatrix(mol)  # Get molecule structure
            # You should extract other features here!
            data = Data(num_nodes=adj.shape[0], edge_index=torch.Tensor(adj).nonzero().T, y=label)
            data_list.append(data)
        self.data, self.slices = self.collate(data_list)
        torch.save((self.data, self.slices), self.processed_paths[0])
