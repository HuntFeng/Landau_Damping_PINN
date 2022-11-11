import numpy as np
import os
import shutil

def clear_datadir(datadir:str):
    if os.path.exists(datadir):
        shutil.rmtree(datadir)
    
    os.mkdir(datadir)