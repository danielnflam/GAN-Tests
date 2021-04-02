import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys, time, datetime


class Dataset_CTCovid19August2020(Dataset):
    """
    TCIA Covid-19 dataset.
    
    Data citation:
    An P, Xu S, Harmon SA, Turkbey EB, Sanford TH, Amalou A, Kassin M, Varble N, Blain M, Anderson V, Patella F, Carrafiello G, Turkbey BT, Wood BJ (2020). CT Images in Covid-19 [Data set]. 
    The Cancer Imaging Archive. DOI: https://doi.org/10.7937/tcia.2020.gqry-nc81
    
    https://wiki.cancerimagingarchive.net/display/Public/CT+Images+in+COVID-19#70227107171ba531fc374829b21d3647e95f532c
    
    TCIA citation:
    Clark K, Vendt B, Smith K, Freymann J, Kirby J, Koppel P, Moore S, Phillips S, Maffitt D, Pringle M, Tarbox L, Prior F. 
    The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository, 
    Journal of Digital Imaging, Volume 26, Number 6, December, 2013, pp 1045-1057. DOI: 10.1007/s10278-013-9622-7.
    """
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Inputs:
            csv_file: Path to CSV file containing patient image names
            root_dir: Directory with all the images.
            transform: Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.images_dataframe = pd.read_csv(os.path.join(csv_file))
        self.transform = transform
        
    def __len__(self):
        return len(self.images_dataframe)
    
    def __getitem__(self, idx):
        """Describe the reading of images in here"""
        if torch.is_tensor(idx):
            idx = idx.tolist() # transform into python list
        
        # Integer indexing
        source_name = os.path.join(self.root_dir, self.images_dataframe.iloc[idx, 1])
        boneless_name = os.path.join(self.root_dir, self.images_dataframe.iloc[idx, 2])
        lung_name = os.path.join(self.root_dir, self.images_dataframe.iloc[idx, 3])
        
        source = np.load(source_name)
        boneless = np.load(boneless_name)
        lung = np.load(lung_name)
        
        sample = {'source': source, 'lung': lung, 'boneless': boneless}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample