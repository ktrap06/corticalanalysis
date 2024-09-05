#!/usr/bin/env python
# coding: utf-8

# # functions_lick_rates 

# In[61]:


import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
from PIL import Image
import plotly.express as px
import imagecodecs
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import filedialog
import os
import re

def licked_txt(file_path, stage):
    if stage == "stage1":
        footer = 3
    else:
        footer = 4
        
    total_lines = sum(1 for line in open(file_path)) # Determine the total number of lines in the file

    data_txt = np.genfromtxt(file_path, delimiter='\t', skip_footer=footer)  # Read the text file into a NumPy array, excluding the last 3 rows

    licked = data_txt[:, 4]
    flash_type = data_txt[:, 2]

    success = []
    if stage == "stage3":
        success = data_txt[:, 5]

    return licked, data_txt, flash_type, success


# In[63]:

def load_files_from_folder(folder_path, stage):
    files = os.listdir(folder_path)
    file_data = {}
    for file_name in files:
        # Extract day number from the file name using regular expressions
        match = re.search(r'day(\d+)', file_name)
        if match:
            day_number = int(match.group(1))
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                print("Loading file:", file_path)
                # Read file data and store it in a dictionary with variable names
                file_data[f'day{day_number}'] = licked_txt(file_path, stage)
    return file_data

# In[64]:


def select_folder(stage):
    folder_path = filedialog.askdirectory()
    if folder_path:
        print("Selected folder:", folder_path)
        file_data = load_files_from_folder(folder_path, stage)
        print("Processing complete.")
        # Example: Access data for day 1: file_data['day1']
        # Example: Access data for day 2: file_data['day2']
    return file_data, folder_path


# In[67]:

def licking(file_data, stage):
    variable_data = []
    
    for idx, data in file_data.items():
        count_1 = 0
        count_2 = 0
        count_3 = 0
        count_4 = 0
        day_number = int(idx[3:])
        licked_values = data[0]
        flash_type_values = data[2]
        success = data[3]
        for value in flash_type_values:
            if value == 1:
                count_1 += 1
            elif value == 2:
                count_2 += 1
            elif value == 3:
                count_3 += 1
            elif value == 4:
                count_4 += 1
    
        if stage == "stage2":
            left_lick = ((sum(licked_values[i] for i in range(len(licked_values)) if flash_type_values[i] == 1))/count_1)*100
            right_lick = ((sum(licked_values[i] for i in range(len(licked_values)) if flash_type_values[i] == 2))/count_2)*100
            left_nogo = ((sum(licked_values[i] for i in range(len(licked_values)) if flash_type_values[i] == 3))/count_3)*100
            right_nogo = ((sum(licked_values[i] for i in range(len(licked_values)) if flash_type_values[i] == 4))/count_4)*100
            total = (((left_lick*count_1)/100+(right_lick*count_2)/100)/(count_1+count_2))*100
            total_nogo = (((left_nogo*count_3)/100+(right_nogo*count_4)/100)/(count_3+count_4))*100
            variable_data.append([day_number, left_lick, right_lick, total, left_nogo, right_nogo,total_nogo])
            print(f"Stage 2 day {day_number} reached left lick rate of {left_lick}% successful, and right lick rate of {right_lick}% successful")
            print(f"Stage 2 day {day_number} reached left nogo rate of {left_nogo}% successful, and right nogo rate of {right_nogo}% successful")
            print(f"total successful trials {total}")
            print(f"total successful nogo trials {total_nogo}")

        elif stage == "stage1": 
            summed_value = sum(data[0])
            variable_data.append([day_number, summed_value])
            print(f"Stage 1 day {day_number} reached a lick rate of {summed_value}%")
            #print(f"total successful trials {total}")
        elif stage == "stage3":
            left_lick = ((sum(licked_values[i] for i in range(len(licked_values)) if flash_type_values[i] == 1))/count_1)*100
            right_lick = ((sum(licked_values[i] for i in range(len(licked_values)) if flash_type_values[i] == 2))/count_2)*100
            
            L_Rlick = ((sum(licked_values[i]+1 for i in range(len(licked_values)) if success[i] == 4))/count_1)*100
            R_Llick = ((sum(licked_values[i]+1 for i in range(len(licked_values)) if success[i] == 3))/count_2)*100

            L_nolick= ((sum(licked_values[i]+1 for i in range(len(licked_values)) if success[i] == 0))/count_1)*100
            R_nolick = ((sum(licked_values[i]+1 for i in range(len(licked_values)) if success[i] == 0))/count_2)*100

            total = (((left_lick*count_1)/100+(right_lick*count_2)/100)/(len(licked_values)))*100

            
            print(f"Stage 3 day {day_number} success left: {left_lick}%, success right:{right_lick}%, fail: left stim right lick")
            print(f"total successful trials {total}")


            variable_data.append([day_number, left_lick, right_lick, L_Rlick, R_Llick, L_nolick, R_nolick, total])
    
    return variable_data


def plot_array(array, stage, export_path):
    # Extract x and y values from the array
    x_values = [row[0] for row in array]
    if stage == "stage1":
        y_values = [row[1] for row in array]
    else:
        y_values_left = [row[1] for row in array]
        y_values_right = [row[2] for row in array]

    # Plot the data
    fig = plt.figure(figsize=(4, 3))
    if stage == "stage1":
        fig,plt.plot(x_values, y_values, marker='o', linestyle='-', color='g')
        fig,plt.legend(["Lick rate"])
    else:
        fig,plt.plot(x_values, y_values_left, marker='o', linestyle='-', color= 'c', label='success: Left')
        fig,plt.plot(x_values, y_values_right, marker='o', linestyle='-', color = 'b', label='success: Right')
        fig,plt.legend(["success: Left", "success: Right"])

    threshold = (np.zeros(len(array)))+50 #threshold 
    fig,plt.plot(x_values, threshold, linestyle='dotted', color='y')
    
    fig,plt.xlabel('Day Number')
    fig,plt.ylabel('Percent licked (%)')
    fig,plt.title('Lick rate per day')
    
    # Set y-axis limit and ticks
    y_max = 110  # Set maximum value to 100 with extra space above
    y_ticks = np.linspace(0, 100, num=11)  # Set ticks from 0 to 100 in intervals of 10
    fig,plt.ylim(0, y_max)  # Set y-axis limit
    fig,plt.yticks(y_ticks)
    
    # Set x-axis ticks
    x_ticks = np.linspace(1, len(x_values), num=len(x_values), dtype=int)  # Set ticks based on the number of data points
    fig,plt.xticks(x_ticks)

    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path)  # Save the plot if a save path is provided
    else:
        plt.show()
        
    return fig

def plot_array_nogo(array, stage, export_path):
    # Extract x and y values from the array
    x_values = [row[0] for row in array]
    if stage == "stage1":
        y_values = [row[1] for row in array]
    else:
        y_values_left = [row[4] for row in array]
        y_values_right = [row[5] for row in array]

    # Plot the data
    fig = plt.figure(figsize=(4, 3))
    if stage == "stage1":
        fig,plt.plot(x_values, y_values, marker='o', linestyle='-', color='g')
        fig,plt.legend(["Lick rate"])
    else:
        fig,plt.plot(x_values, y_values_left, marker='o', linestyle='-', color= 'g', label='success: Left')
        fig,plt.plot(x_values, y_values_right, marker='o', linestyle='-', color = 'm', label='success: Right')
        fig,plt.legend(["success: Left nogo", "success: Right nogo"])

    threshold = (np.zeros(len(array)))+50 #threshold 
    fig,plt.plot(x_values, threshold, linestyle='dotted', color='y')
    
    fig,plt.xlabel('Day Number')
    fig,plt.ylabel('Percent licked (%)')
    fig,plt.title('nogo rate per day')
    
    # Set y-axis limit and ticks
    y_max = 110  # Set maximum value to 100 with extra space above
    y_ticks = np.linspace(0, 100, num=11)  # Set ticks from 0 to 100 in intervals of 10
    fig,plt.ylim(0, y_max)  # Set y-axis limit
    fig,plt.yticks(y_ticks)
    
    # Set x-axis ticks
    x_ticks = np.linspace(1, len(x_values), num=len(x_values), dtype=int)  # Set ticks based on the number of data points
    fig,plt.xticks(x_ticks)

    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path)  # Save the plot if a save path is provided
    else:
        plt.show()
        
    return fig

def plot_fail(array, export_path):
    x_values = [row[0] for row in array]
    
    y_L_Rlick = [row[3] for row in array]
    y_R_Llick = [row[4] for row in array]    
        
    y_L_nolick = [row[5] for row in array]
    y_R_nolick = [row[6] for row in array]

    fig = plt.figure(figsize=(4, 3))
    fig,plt.plot(x_values, y_L_Rlick, marker='o', linestyle='-', label='fail:left stim right lick')
    fig,plt.plot(x_values, y_R_Llick, marker='o', linestyle='-', label='fail:right stim left lick')
        
    fig,plt.plot(x_values, y_L_nolick, marker='o', linestyle='-', label='fail: left stim no lick')
    fig,plt.plot(x_values, y_R_nolick, marker='o', linestyle='-', label='Right: right stim no lick')
    fig,plt.legend(["fail:left stim right lick","fail:right stim left lick","fail: left stim no lick","fail:right stim no lick"])

    threshold = (np.zeros(len(array)))+75
    fig,plt.plot(x_values, threshold, linestyle='dotted', color='y')
    
    fig,plt.xlabel('Day Number')
    fig,plt.ylabel('fail licked (%)')
    fig,plt.title('Lick fail rate per day')
    
    # Set y-axis limit and ticks
    y_max = 110  # Set maximum value to 100 with extra space above
    y_ticks = np.linspace(0, 100, num=11)  # Set ticks from 0 to 100 in intervals of 10
    fig,plt.ylim(0, y_max)  # Set y-axis limit
    fig,plt.yticks(y_ticks)
    
    # Set x-axis ticks
    x_ticks = np.linspace(1, len(x_values), num=len(x_values), dtype=int)  # Set ticks based on the number of data points
    fig,plt.xticks(x_ticks)

    plt.tight_layout()
    
    if export_path:
        plt.savefig(export_path)  # Save the plot if a save path is provided
    else:
        plt.show()
        
    return fig

def plot_total(array, stage, export_path_total):
    # Extract x and y values from the array
    x_values = [row[0] for row in array]
    if stage == "stage2":
        y_values = [row[3] for row in array]
    if stage == "stage3":
        y_values = [row[7] for row in array]
    fig = plt.figure(figsize=(4, 3))

    fig,plt.plot(x_values, y_values, marker='o', linestyle='-', color='g')
    fig,plt.legend(["Lick rate"])

    threshold = (np.zeros(len(array)))+75
    fig,plt.plot(x_values, threshold, linestyle='dotted', color='y')
    
    fig,plt.xlabel('Day Number')
    fig,plt.ylabel('Percent licked (%)')
    fig,plt.title('Lick rate per day total number')
    
    # Set y-axis limit and ticks
    y_max = 110  # Set maximum value to 100 with extra space above
    y_ticks = np.linspace(0, 100, num=11)  # Set ticks from 0 to 100 in intervals of 10
    fig,plt.ylim(0, y_max)  # Set y-axis limit
    fig,plt.yticks(y_ticks)
    
    # Set x-axis ticks
    x_ticks = np.linspace(1, len(x_values), num=len(x_values), dtype=int)  # Set ticks based on the number of data points
    fig,plt.xticks(x_ticks)

    plt.tight_layout()
    
    if export_path_total:
        plt.savefig(export_path_total)  # Save the plot if a save path is provided
    else:
        plt.show()
        
        
    return fig

def plot_total_nogo(array, stage, export_path_nogo):
    # Extract x and y values from the array
    x_values = [row[0] for row in array]
    y_values_lick = [row[3] for row in array]
    y_values_nogo = [row[6] for row in array]
    fig = plt.figure(figsize=(4, 3))
    
    fig,plt.plot(x_values, y_values_lick, marker='o', linestyle='-', color= 'r', label='success lick')
    fig,plt.plot(x_values, y_values_nogo, marker='o', linestyle='-', color = 'c', label='success nogo')
    fig,plt.legend(["success lick", "success nogo"])
    
    threshold = (np.zeros(len(array)))+75
    fig,plt.plot(x_values, threshold, linestyle='dotted', color='y')
    
    fig,plt.xlabel('Day Number')
    fig,plt.ylabel('Percent success (%)')
    fig,plt.title('success rate per day total number')
    
    # Set y-axis limit and ticks
    y_max = 110  # Set maximum value to 100 with extra space above
    y_ticks = np.linspace(0, 100, num=11)  # Set ticks from 0 to 100 in intervals of 10
    fig,plt.ylim(0, y_max)  # Set y-axis limit
    fig,plt.yticks(y_ticks)
    
    # Set x-axis ticks
    x_ticks = np.linspace(1, len(x_values), num=len(x_values), dtype=int)  # Set ticks based on the number of data points
    fig,plt.xticks(x_ticks)

    plt.tight_layout()
    
    if export_path_nogo:
        plt.savefig(export_path_nogo)  # Save the plot if a save path is provided
    else:
        plt.show()
        
        
    return fig

def plot_total_pretty(array, stage, export_path_total=None):
    # Extract x and y values from the array
    x_values = [row[0] for row in array]
    if stage == "stage2":
        y_values = [row[3] for row in array]
    elif stage == "stage3":
        y_values = [row[7] for row in array]
    else:
        raise ValueError("Invalid stage. Must be 'stage2' or 'stage3'.")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Change the figure and axes background colors
    fig.patch.set_facecolor('#F3F6F1')
    ax.set_facecolor('#F3F6F1')

    # Plot the data
    ax.plot(x_values, y_values, marker='o', linestyle='-', color='#589370', label='Lick rate', linewidth=2.5)
    # Add the threshold line
    #threshold = np.zeros(len(array)) + 75
    #ax.plot(x_values, threshold, linestyle='dotted', color='y', label='Threshold')

    # Add labels and title
    ax.set_xlabel('Day Number')
    ax.set_ylabel('Percent Licked (%)')
    ax.set_title('HD Stage 3', fontsize=18)
    
    # Set y-axis limit and ticks
    y_max = 110
    y_ticks = np.linspace(0, 100, num=6)
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.spines['top'].set_visible(False) #hides the top spine.
    ax.spines['right'].set_visible(False) #hides the right spine.
    
    # Set x-axis ticks
    x_ticks = np.linspace(1, len(x_values), num=9, dtype=int)  # Set ticks based on the number of data points
    ax.set_xticks(x_ticks)
    ax.set_xlabel('Day Number', fontsize=16)
    ax.set_ylabel('Percent Success (%)', fontsize=16)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    # Add legend and grid
    #ax.legend()
    #ax.grid(False, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    if export_path_total:
        plt.savefig(export_path_total)  # Save the plot if a save path is provided
    else:
        plt.show()
        
    return fig

    