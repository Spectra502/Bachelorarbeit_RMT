from typing import Dict,List
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator


import torch
import torch.nn as nn
import torch.nn as F
import torch.utils.data as data
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, random_split
from torch import rand, randint, Tensor

class Group_Grat_Dataset(Dataset):
  #data loading
  def __init__(self, paths, labels):
    self.paths = paths
    self.labels = labels

  #len(dataset)
  def __len__(self):
    return len(self.paths)

  #dataset [0]
  def __getitem__(self, idx):
    path = self.paths[idx]
    label = self.labels[idx]

    return path, label

def extractFSWDataset(FSW_ZIP_PATH, FSW_working_dir_name):
    """
    This function 
    Parameters:
        FSW_zip_path (string): Path of the FSW zip file
        FSW_working_dir_name (string): Name of folder where the FSW zipfile will be extracted
    Returns:
        
    Example:
    """
    HX_TRAINING_ORIGIN_DIR_PATH = FSW_ZIP_PATH
    DATA_DIR_PATH = Path(f"./{FSW_working_dir_name}")
    
    #creates working directory
    if os.path.exists(DATA_DIR_PATH):
        print(f"{DATA_DIR_PATH} directory already exists")
    else:
        DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)
    
    #copies hx_training_directory to working directory
    try:
      HX_TRAINING_DEST_DIR_PATH = os.path.join(DATA_DIR_PATH,"hx_training")
      shutil.copytree(HX_TRAINING_ORIGIN_DIR_PATH, HX_TRAINING_DEST_DIR_PATH)
    except:
      print("Already copied")
    
    #unzips image folder
    try:
      ZIPFILE_PATH = os.path.join(HX_TRAINING_DEST_DIR_PATH, "hx_training_classify.zip")
      with zipfile.ZipFile(ZIPFILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR_PATH)
    except:
      print("data alredy unziped")

    return HX_TRAINING_DEST_DIR_PATH

def orderFSWDataset(FSW_DATA_PATH, delete_unziped_data):
    #transforms CSV file to a DataFrame and a Dictionary
    ANNOTATIONS_PATH = Path(f"{FSW_DATA_PATH}/hx_training/FSW dataset A_annotations.csv")
    if os.path.exists(ANNOTATIONS_PATH):
        dataframe = pd.read_csv(ANNOTATIONS_PATH, sep=";")
        dataframe['img_path'] = dataframe['img_path'].str.replace('..','.',1)
        
        # Copy hx_training_directories to data/raw directory
        try:
          SOURCE_DIR = Path('./data/hx_training_classify')  # /content/data/hx_training_classify
          DEST_DIR = './data/raw/hx_fsw/images'
          shutil.copytree(SOURCE_DIR, DEST_DIR)
        except:
          print("path already exists")
        
        #deletes unziped dir
        shutil.rmtree(f"{FSW_DATA_PATH}/hx_training_classify") if delete_unziped_data else print("Unziped data still stored")
    else:
        break
    return dataframe
    
"""
def dataframeColumnSelectAndOrderByGroups(dataframeFSW, selectedLabel):
    selected_columns = ['img_path', selectedLabel, 'Weld Seam Nr.', 'Group']
    dataframe = dataframeFSW[selected_columns]
    
    #sorts data frame by groups
    dataframe_groups = {
        "group_1": dataframe_group_1 = dataframe[dataframe['Group'] == 1],
        "group_2": dataframe_group_2 = dataframe[dataframe['Group'] == 2].reset_index(drop=True),
        "group_3": dataframe_group_3 = dataframe[dataframe['Group'] == 3].reset_index(drop=True),
        "group_4": dataframe_group_4 = dataframe[dataframe['Group'] == 4].reset_index(drop=True),
        "group_5": dataframe_group_5 = dataframe[dataframe['Group'] == 5].reset_index(drop=True),
        "group_6": dataframe_group_6 = dataframe[dataframe['Group'] == 6].reset_index(drop=True),
        "group_7": dataframe_group_7 = dataframe[dataframe['Group'] == 7].reset_index(drop=True)
    }

    return dataframe_groups
"""

def createGroupDatasets(dataframe_groups, train_percentage, eval_percentage, test_percentage):
    group_df = {}
    for group, dataframe_group in dataframe_groups.items():
        group_df[f"{group}_df"] = dataframe_groups[group][['img_path', 'grat_label']]
    group_datasets = {}
    i = 1
    for group, dataframe in group_df.items():
        group_datasets[f"group_{i}_dataset"] = Group_Grat_Dataset(paths=dataframe['img_path'], labels=dataframe['grat_label'])
    
    
    
    

    