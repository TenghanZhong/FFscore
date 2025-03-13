import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats  #t检验 from scipy import stats
import statsmodels.api as sm
import warnings
import os
import glob
import tushare as ts
from tqdm import tqdm
import re
import sys
from multiprocessing import Pool
warnings.filterwarnings("ignore")
directory ='D:\\Stock_data_fscore'
csv_files = glob.glob(os.path.join(directory, '*.csv.gz'))
for file_path in csv_files:
    try:
        # Load the content of the file into a pandas DataFrame
        df = pd.read_csv(file_path, compression='gzip')
        # Check if the DataFrame is empty (only headers, no data)
        if df.empty:
            # If empty, delete the file
            os.remove(file_path)
            print(f"Deleted empty file: {file_path}")
    except Exception as e:
        # In case of an error (e.g., file is corrupted, not found, etc.), print the error message
        print(f"Error processing file {file_path}: {e}")