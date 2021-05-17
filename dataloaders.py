import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os, sys, time, datetime
import pydicom.fileset
import matplotlib.pyplot as plt

###########################
# JSRT CXR dataset
# Shiraishi J, Katsuragawa S, Ikezoe J, Matsumoto T, Kobayashi T, Komatsu K, Matsui M, Fujita H, Kodera Y, and Doi K.: Development of a digital image database for chest radiographs with and without a lung nodule: Receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. AJR 174; 71-74, 2000
###########################
class JSRT_CXR(Dataset):
    def __init__(self, data_normal, data_BSE, transform):
        """
        JSRT Chest X-ray dataset.
        Source:
        Shiraishi J, Katsuragawa S, Ikezoe J, Matsumoto T, Kobayashi T, Komatsu K, Matsui M, Fujita H, Kodera Y, and Doi K.
        Development of a digital image database for chest radiographs with and without a lung nodule: Receiver operating characteristic analysis of radiologists’ detection of pulmonary nodules. 
        AJR 174; 71-74, 2000
        
        Inputs:
            data_normal: root directory holding the normal / non-suppressed images
            data_BSE: root directory holding the bone-suppressed images
            transform: (optional) a torchvision.transforms.Compose series of transformations
        Assumed that files corresponding to the same patient have the same name in both folders data_normal and data_BSE.
        """
        sample = {"Patient": [], "boneless":[], "source":[]}
        for root, dirs, files in os.walk(data_BSE):
            for name in files:
                if '.png' in name:
                    a_filepath = os.path.join(root, name)
                    # Patient code
                    head, tail = os.path.split(a_filepath)
                    patient_code_file = os.path.splitext(tail)[0]
                    # Place into lists
                    sample["Patient"].append(patient_code_file)
                    sample["boneless"].append(a_filepath)

                    # For each patient code, search the alternate data_folder to obtain the corresponding source
                    for root2, dirs2, files2 in os.walk(data_normal):
                        for name2 in files2:
                            # Need regex to distinguish between e.g. 0_1 and 0_10
                            filename2,_ = os.path.splitext(name2)
                            if patient_code_file == filename2:
                                sample["source"].append(os.path.join(root2, name2))

        self.data = pd.DataFrame(sample)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Describe the reading of images in here"""
        if torch.is_tensor(idx):
            idx = idx.tolist() # transform into python list
        
        patient_code = self.data["Patient"].iloc[idx]
        source_image = plt.imread(self.data["source"].iloc[idx])
        boneless_image = plt.imread(self.data["boneless"].iloc[idx])
        source_image, boneless_image = self._check_if_array_3D(source_image, boneless_image)
        
        sample = {'source': source_image, 'boneless': boneless_image} #'patientCode': patient_code
        
        if self.transform:
            sample = self.transform(sample)
        return sample
        
    def visualise(self, idx):
        bonelessIm = plt.imread(self.data["boneless"].iloc[idx])
        sourceIm = plt.imread(self.data["source"].iloc[idx])
        sourceIm, bonelessIm = self._check_if_array_3D( sourceIm, bonelessIm)
        
        # Visualisation
        fig, ax=plt.subplots(1,2)
        ax[0].imshow(sourceIm, cmap="gray")
        ax[1].imshow(bonelessIm, cmap="gray")
    
    # Helper function
    def _check_if_array_3D(self, source_image, boneless_image):
        # Check if array is 3D or 2D
        iters = 0
        for image in [source_image, boneless_image]:
            if image.ndim == 3:
                # make the image grayscale
                image = image[:,:,0]
            iters+=1
            if iters == 1:
                source_image = image
            if iters == 2:
                boneless_image = image
        return source_image, boneless_image
    
    
#####################################
# CT Datasets
#####################################
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