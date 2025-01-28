#!/usr/bin/env python
# coding: utf-8

# # heatmaps functions

import numpy as np
import matplotlib.pyplot as plt
import cv2
import plotly.express as px
import os
import pickle
import seaborn as sns
from scipy.stats import zscore

def load_pickle(condition, flash, success_rate, days, length = 200):
    #condition = WT_mice or HD_mice (as a list, make sure to define this first
    #flash = "single", "double", or "nogo"
    #success_rate = "success" or "fail"
    #days = trial days (1-15) as a list, define this first
    #length = duration of trace (how many frames)

    dataset = []
    for mouse in condition:
        one_mouse = []
        dataset.append(one_mouse)
        for day in days:
            path = f"{shared_path}/{flash}/{success_rate}/full_frame/{mouse}_stage2_{day}_{flash}_{success_rate}_ff.pickle"
            #print(path)
            try:
                with open(path, 'rb') as file:
                    one_day = pickle.load(file)
                one_mouse.append(one_day)
            except FileNotFoundError:
                # If the file is not found, append NaN of a predefined length
                default_length = length  # or another length depending on your dataset
                one_mouse.append(np.full(default_length, np.nan))  # Append NaNs
                print(f"File not found for mouse {mouse} on day {day}")
    return dataset

