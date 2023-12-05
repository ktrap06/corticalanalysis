#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""install the packages to run this code"""
#%pip install matplotlib tifffile scipy tqdm pybaselines


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
import tifffile
#import cv2
#from PIL import Image
#from pybaselines.whittaker import asls
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from scipy.signal import cheby1, filtfilt, find_peaks, peak_widths
from tqdm import tqdm


# In[11]:


video_path = "//files.ubc.ca/team/BNRC/NINC/Raymond Lab/Kaiiiii/cortical data/Pilot/2023_11_01/405700_f3_stage2_pilot_day3_violet.tif"


# In[27]:


def load_frames(video_path):
    """
    Loads in tif stacks as a 3d array. Make sure the dtype is 12.
    :param video_path: path to the file, including file name.tif

    :return: 3d array of 128x128 x timeseries
    """

    print("video loading please wait...")
    frames = tifffile.imread(video_path)
    print("dimensions are", frames.shape, "as a", frames.dtype)
    return frames


# In[28]:


load_frames(video_path);


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




