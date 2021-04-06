import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys, time, datetime
import pydicom.fileset

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
        Outputs:
            sample (dict) with keys:
                'source': (ndarray) the original DRR generated from the CT image
                'boneless': (ndarray) the DRR generated with bone intensities = 0
                'lung': (ndarray) the DRR generated with the lung segment only
                'PixelSize': (tuple) the pixel dimensions in (rows, columns) in milimetres
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
        
        sample = {'source': source, 'lung': lung, 'boneless': boneless, 'PixelSize': (self.images_dataframe.iloc[idx,4],self.images_dataframe.iloc[idx,5])}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
class Dataset_PolyU_CXR(Dataset):
    """
    CXR data from PolyU -- this is used as conditioning data for the generator & discriminator.
    Generate a dataframe in memory to hold the details, but NOT the images.
    """
    def __init__(self, root_dir, transform=None):
        """
        Construct the dataframe here.
        """
        
        path_to_file = []
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                if name=="DICOMDIR":
                    path_to_file.append(os.path.join(root, name))
        
        # Lists to create a dataframe table
        PatientID = []
        StudyDate = []
        StudyTime = []
        Modality = []
        SOPClassUID = []
        SOPInstanceUID = []
        # Select a CXR directory
        for path_idx in range(len(path_to_file)):
            a_path = path_to_file[path_idx]
            fs = pydicom.fileset.FileSet(a_path)
            
            # Select an instance inside this directory
            # Check whether it is a CR / DX image
            for instance in fs:
                if "1.2.840.10008.5.1.4.1.1.1" in instance.SOPClassUID:  # check if CR or DX image
                    PatientID.append(instance.PatientID)
                    StudyDate.append(instance.StudyDate)
                    StudyTime.append(instance.StudyTime)
                    Modality.append(instance.Modality)
                    SOPInstanceUID.append(instance.SOPInstanceUID)
                    SOPClassUID.append(instance.SOPClassUID)
        
        # Make dataframe
        data = {
            'PatientID': PatientID,
            'StudyDate': StudyDate,
            'StudyTime': StudyTime,
            'Modality': Modality,
            'SOPClassUID': SOPClassUID,
            'SOPInstanceUID': SOPInstanceUID
        }
        df = pd.DataFrame(data)
        self.images_dataframe=df
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.images_dataframe)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist() # transform into python list
        
        # Take a look at the CXR dataframe
        PatientID = self.images_dataframe.iloc[idx,0]
        path_to_DICOMDIR = os.path.join(self.root_dir , PatientID , "DICOMDIR")
        fs = pydicom.fileset.FileSet(path_to_DICOMDIR)
        for instance in fs:
            if instance.SOPInstanceUID == self.images_dataframe.iloc[idx, 5]:
                ds = instance.load()
                PixelSpacing = ds.ImagerPixelSpacing
                image = ds.pixel_array
        
        sample = {'source': image, 'PixelSize': PixelSpacing}
        if self.transform:
            sample = self.transform(sample)
        return sample