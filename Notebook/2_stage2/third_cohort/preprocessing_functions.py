#!/usr/bin/env python
# coding: utf-8

# In[4]:


"""install the packages to run this code"""
#%pip install matplotlib tifffile scipy tqdm pybaselines opencv-python imagecodecs #plotly #need to do this through conda install


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import tifffile
import cv2
from PIL import Image
#from pybaselines.whittaker import asls
import scipy 
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter
from scipy.signal import cheby1, filtfilt, find_peaks, peak_widths
from tqdm import tqdm
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
import imagecodecs
from matplotlib.animation import FuncAnimation
import seaborn as sns



# In[6]:


def load_frames(video):
    """
    Loads in tif stacks as a 3d array. Make sure the dtype is 12.
    :param video: path to the file, including file name.tif

    :return: 3d array of 128x128 x timeseries
    """

    print("loading video, please wait...")
    frames = tifffile.imread(video)
    print("dimensions are", frames.shape, "as a", frames.dtype)
    return frames


# In[7]:


def interactive_plot(array1d, title_label = "output", xaxis_label = "duration", yaxis_label = "intensity"):
    """make plotly interactive plots"""
    fig = px.line(array1d)
    fig.update_layout(
    title= title_label,
    xaxis=dict(title=xaxis_label),
    yaxis=dict(title=yaxis_label),
    xaxis_rangeslider_visible=True,
    showlegend=False)
    fig.show()


# In[8]:


def plot_frame(frames:np.ndarray, frame_number, cmap = 'jet', color_label = "N/A"):
    """
    plots a frame of interest, to check if the video was loaded in properly
    :param frames: 3d array loaded in with tifffile
    :param frame_number: frame of interest to plot
    :param cmap: colormap used, set to jet as default
    """

    image = frames[frame_number, :, :]
    fig, ax = plt.subplots()
    
    colormap = plt.get_cmap(cmap)
    colormap.set_bad('black', alpha=1.0) #make it so nan values (in the mask) are black
    
    im = ax.imshow(image, cmap=cmap)
    plt.colorbar(im, label = color_label)
    
    ax.grid(True)
    ax.axis('on')
    print("frame number", frame_number, "plotted")
    plt.show()


# In[9]:


def temporal_mean(frames):
    """
    calculate and plot the signal over time
    
    :param frames: 3d array loaded in with tifffile
    :return: 1d array where each frame intensity is averaged
    """
    
    mean_timecourse = frames.mean(axis=(1,2))

    min_value = np.min(mean_timecourse)     # Find the minimum value in the signal array
    dark_frame_threshold = min_value + 0.2 * min_value     # Calculate the threshold as the minimum value plus 10%
    indices = np.arange(len(frames))
    
    #plotting
    print("default 20% threshold is", dark_frame_threshold, "Determine if adjustments need to be made")
    interactive_plot(mean_timecourse, title_label = "raw mean timecourse", xaxis_label = "frame number (30 fps)", yaxis_label = "intensity value (A.U)")

    return mean_timecourse


# In[10]:


def remove_dark_frames(frames, signal, threshold = 0.2):
    """
    calculate where the dark frames are at the start and end of the trial

    :param frames: 3d array loaded in with tifffile
    :param signal: 1d array mean_time course array
    :param threshold: a percent above the minimum value, default set to 20%

    return: 
    """
    
    min_value = np.min(signal)     # Find the minimum value in the signal array
    dark_frame_threshold = min_value + threshold * min_value     # Calculate the threshold as the minimum value plus 10%
    
    brain_indices = np.where(signal > dark_frame_threshold)[0]
    start_index = brain_indices[0]+1
    end_index = brain_indices[-1]
    
    frames = frames[start_index:end_index, :, :]
    signal = signal[start_index:end_index]

    #plotting
    print("threshold determined to be:", dark_frame_threshold)
    interactive_plot(signal, title_label = "mean timecourse with dark frames removed", xaxis_label = "frame number (30 fps)", yaxis_label = "intensity value (A.U)")
    
    return signal, frames


# In[11]:


def extract_artifacts(timecourse, frames_between_double_flash = 33):
    """
    extract indices where the artifact flash exists, as a function of 3 times the standard deviation of the global mean
    :param timecourse: 1d array of the mean time course
    :param frames_between_double_flash: the set time between two flashes in the double flash condition, default set to 20, always estimate higher

    :return: array of where the signal is above the mean, artifact flashes
    """

    T_mean = timecourse.mean()
    std = timecourse.std()
    artifact_indices = find_peaks(timecourse, height=(T_mean + 3 * std), distance=10)[0] #indescriminatly finds all the flashes

    """ THIS PART IS FOR USE WHEN NEEDING TO SEPARATE SINGLE AND DOUBLE"""
    time_diff = np.diff(artifact_indices) #determine the time between each peak, if it is 15 frames it is double, 320 frames is single

    single_flash = []
    double_flash = []

    for i in range(len(artifact_indices)):
        if  (i > 0 and artifact_indices[i] - artifact_indices[i - 1] < frames_between_double_flash):
            double_flash.append(artifact_indices[i])
        elif (i < len(artifact_indices) - 1 and artifact_indices[i + 1] - artifact_indices[i] < frames_between_double_flash):
            double_flash.append(artifact_indices[i])
        else:
            single_flash.append(artifact_indices[i])

    print("Number of double_flash:", len(double_flash))

    artifact_indices = np.array(artifact_indices)
    double_flash = np.array(double_flash).reshape(-1, 2)
    single_flash = np.array(single_flash)

    print("Number of double_flash pairs:", len(double_flash))
    print("Number of single_flash:", len(single_flash))
    print("Number of total flashes:", len(artifact_indices))

    """SECTION END"""
    
    return artifact_indices ,single_flash, double_flash


# In[12]:


def interpolate(no_dark_frames, artifact_indices, stim_length):
    """
    interpolate over the artifacts, representing a total of 5 frames. 

    :param frames: 3d array with dark frames removed, DO NOT USE ORIGINAL "FRAMES" (includes each pixel value over time)
    :param artifact_indices: index of where the artifact are located in the frames array
    

    :return: 
    """
    
    print("interpolating artifact frames...")
    interp_indices = []        #make the indices that need to be interpolated, organized in groups of 5
    for index in artifact_indices:
        start = index-stim_length        #add 3 frame padding before, to ensure the entire artifact is removed
        end = index+stim_length          #add 3 frame padding after, to ensure the entire artifact is removed
        group = []
        for new_index in range(start, end+1):        #loop within to ensure each array within the array is a total of 7 (for each flash of light)
            group.append(new_index)
        interp_indices.append(group)

    x = [0, 6]                                                                  #between the two interpolated side, split into equaL segments
    xnew = np.linspace(0, 6, np.shape(interp_indices)[1])                       #linear interpolate between the two x values, at equal increments
    for g in tqdm(range(len(interp_indices[:]))):                                                 #tqdm is loading package (GUI), loops 3 dimensions of array
        y_2d = [no_dark_frames[interp_indices[g][0]], no_dark_frames[interp_indices[g][-1]]]      #the frames which are usable on either side of artifact
        for r in range(no_dark_frames.shape[1]):                                                  #iterate over each row of pixels
            for c in range(no_dark_frames.shape[2]):                                              #iterate over each column of pixels
                y = [y_2d[0][r, c], y_2d[1][r, c]]                              #represents each pixel [r,c], and the first and last frame to interpolate between
                f = interp1d(x, y)                                              #interpolate the y values between these two-time segments
                ynew = f(xnew)
                no_dark_frames[interp_indices[g], r, c] = ynew[:]

    #plot the interpolation
    interactive_plot(no_dark_frames.mean(axis=(1,2)), title_label = "mean timecourse with without artifacts", xaxis_label = "frame number (30 fps)", yaxis_label = "intensity value (A.U)")
    
    return no_dark_frames


# In[13]:


def dff(data, duration = 10, frame_rate = 30, eps=1e-3):
    """
     Calculate dF/F (moving centered mean) for tif video data.

    :param data: 3d array with shape (depth, height, width), use the no artifact array
    :param duration: Duration over which to average in seconds, default set at 10
    :param frame_rate: Frame rate of the data in frames per second, default set at 30fps

    :return: 3d array including the padding on both ends
    """
    num_frames = int(duration * frame_rate)
    
    pad_size = int(num_frames / 2)
    depth, height, width = data.shape[0], data.shape[1], data.shape[2]          # initialize the result array
    dFF = np.ones((depth + 2 * pad_size, height, width), dtype=np.float32)
    centered_mean_array = np.zeros_like(data)

    data = np.concatenate((np.ones((pad_size, height, width)), data), axis=0)  # add padding to the data array (required for mean calculation)
    data = np.concatenate((data, (np.ones((pad_size, height, width)))), axis=0)
    for h in tqdm(range(height)):
        for w in range(width):
            data[0:pad_size,h,w] = data[pad_size:pad_size*2,h,w].mean()      #for each pixel (h=height, w=width), replace the start padding with the mean of the first 10 seconds
            data[-pad_size:,h,w] = data[-pad_size*2:-pad_size,h,w].mean()

    # iterate over frames starting from num_frames
    for t in tqdm(range(pad_size, depth+pad_size)):        # t starts at pad_size (the adjusted index 0 after padding)
        current_frame = data[t, :, :]        # get the current frame
        centered_mean = np.mean(data[(t - pad_size):(t + pad_size), :, :], axis=0)       # calculate the centered mean with respect to the current frame (t)
        centered_mean_array[t-pad_size,:,:] = centered_mean       #need subtract pad_size because the centered_mean_array does not include the padding      
        dF = current_frame - centered_mean               # calculate dF
        dFF[t, :, :] = (dF / (centered_mean + eps))              # calculate dF/F and store in the result array

    return dFF[pad_size:-pad_size, :, :], centered_mean_array


# In[14]:


def smoothing(data):
    """
    Smooth the data using a Gaussian filter for the spatial dimension and
    Chebyshev filter for the temporal dimension.

    :param data: The dff 3d array with shape (depth, height, width), dff signal
    
    :return: Normalized 3d array with shape (depth, height, width) with spatial and temporal smoothing
    """
    
    #spatial smoothing
    data =  gaussian_filter(data, sigma=2.0, radius=3)

    #temporal smoothing with t
    fs = 30.0  # sampling frequency
    Ny = 0.5 * fs  # Nyquist frequency

    lp = 0.1  # lower bound of the desired frequency band
    hp = Ny - 0.001  # upper bound of the desired frequency band

    # design the Chebyshev type I filter (see SciPy documentation for details)
    chebyshev = cheby1(N=2,    #2nd Order
                       rp=0.5,    #max ripple allowed below unity gain in the passband
                       Wn=[lp / Ny, hp / Ny], #Wn is in half-cycles / sample
                       btype='band',
                       analog=False,
                       output='ba')     #backwards compatibility

     # initialize the result array
    result = np.empty_like(data)
    depth, height, width = data.shape[0], data.shape[1], data.shape[2]


    # apply the filter along the time axis to each pixel (temporal dimension)
    for h in tqdm(range(height)):
        for w in range(width):
            result[:, h, w] = filtfilt(b=chebyshev[0], a=chebyshev[1], x=data[:, h, w])

    # normalize the result
    result = (result - result.min()) / (result.max() - result.min())
    np.clip(result, 0, 1)  # clip values to be between 0 and 1
    
    return data


# In[15]:


def save_tiff(array3d, output_stack_path="X:/Raymond Lab/Kaiiiii/processed_data/dff_output.tiff"):
    """
    saves the tiff stack as a 32 bit array, notably not reduced to 8-bit

    :param array3d: use the smoothed data in a 3d numpy array
    :param mask_path: path where to save the output
    """
    
    array3d = array3d.astype(np.float32)
    individual_slices = []  # Initialize an empty list to store individual slices

    for i in tqdm(range(len(array3d))):  # Iterate through the depth dimension
        image_data = array3d[i].astype(np.float32)  # Convert the data to float32 without scaling
        individual_slices.append(image_data)  # Append the image data to the list

    stacked_data = np.array(individual_slices)  # Convert the list to a NumPy array

    # Save the 3D array as a TIFF stack
    tifffile.imwrite(output_stack_path, stacked_data)

    print("Combined TIFF stack saved successfully.")


# In[16]:


def load_txt(file_path, fps=30, footer = 4):
    total_lines = sum(1 for line in open(file_path)) # Determine the total number of lines in the file

    data_txt = np.genfromtxt(file_path, delimiter='\t', skip_footer=footer)  # Read the text file into a NumPy array, excluding the last 3 rows
    data_footer = np.genfromtxt(file_path, delimiter='\t', skip_header=(total_lines-footer), filling_values=np.nan)  #load dark frames to line up data
    start, end = data_footer[0,1], data_footer[1,1 ] #start is LED on time, end is LED off time

    time_column = data_txt[:, 1]
    flash_type = data_txt[:, 2]
    frames_flash_index = (time_column - start)*fps

    return frames_flash_index, data_txt, flash_type


# In[17]:


def extract_txt_flashes(txt_flashes, flash_type, fps = 30, flash_interval = 0.5):
    allflashes = []
    double = []
    single = []
    for i, f in enumerate(txt_flashes):
        if flash_type[i] == 2:
            allflashes.append(f)
            allflashes.append(f+(flash_interval*fps))
            double.append(f)
            double.append(f+(flash_interval*fps))
        else:
            allflashes.append(f)
            single.append(f)
    allflashes = np.array(allflashes).round()
    single = np.array(single).round()
    double = np.array(double).round().reshape(-1, 2)

    return allflashes, single, double


# In[18]:


def load_masks(mask_path = 'C:/Users/trapped/Documents/GitHub/corticalanalysis/mouse_atlas.png', mask_outline = 'C:/Users/trapped/Documents/GitHub/corticalanalysis/mouse_atlas_outline.png'):
    """
    applies a png image of a allen mouse brain atlas of the functional regions over the tiff stack

    :param array3d: use the smoothed data in a 3d numpy array
    :param mask_path: path to the atlas png

    :return: a stack with the areas which do not represent the brain areas set as zeros
    """
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Load PNG mask
    mask = cv2.resize(mask, (128, 128)) # Resize mask to match TIFF stack dimensions if needed. Assuming your TIFF stack has dimensions (num_frames, 128, 128)
    #_, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY) # Normalize mask to binary (0 or 255)


    outline =  cv2.imread(mask_outline, cv2.IMREAD_GRAYSCALE) # Load PNG mask
    outline = cv2.resize(outline, (128, 128)) # Resize mask to match TIFF stack dimensions if needed. Assuming your TIFF stack has dimensions (num_frames, 128, 128)
    #_, binary_mask = cv2.threshold(outline, 128, 255, cv2.THRESH_BINARY) # Normalize mask to binary (0 or 255)

    return mask, outline

#roi processing

def avg_trials(flash_indices, smoothed_signal, double=False, segment_size=330, minus=0):
    segments = []
    
    if double:
        for flash_index in flash_indices[:,0]:
            d_start_index = flash_index - minus
            d_end_index = d_start_index + segment_size
            segment = smoothed_signal[d_start_index:d_end_index]
            if len(segment) == segment_size:
                segments.append(segment)
            else:
                print(f"Segment double length mismatch: expected {segment_size}, got {len(segment)}")
    else:
        for flash_index in flash_indices:
            start_index = flash_index - minus
            end_index = start_index + segment_size - minus
            segment = smoothed_signal[start_index:end_index]
            if len(segment) == segment_size:
                segments.append(segment)
            else:
                print(f"Segment single length mismatch: expected {segment_size}, got {len(segment)}")
                
    if segments:
        segments_array = np.array(segments)
        mean_signal = np.mean(segments_array, axis=0)
        return mean_signal, segments_array
    else:
        print("No valid segments found")
        return None

def extract_v1(avg_flash, roi_time = 5, roi_start_x=15, roi_start_y = 85, roi_width=45, roi_height = 45):
    
    #Determine the coordinates of the first region (ROI)
    first_region = avg_flash[roi_time, roi_start_y:roi_start_y+roi_height, roi_start_x:roi_start_x+roi_width]
    
    #Find the coordinates of the maximum value within the extracted region
    max_coords = np.unravel_index(np.argmax(first_region), first_region.shape)
    max_x, max_y = max_coords[1] + roi_start_x, max_coords[0] + roi_start_y

    new_roi_width, new_roi_height = 10, 10  # Adjust the dimensions of the new region
    new_roi_start_x = max_x - new_roi_width // 2
    new_roi_start_y = max_y - new_roi_height // 2
    
    # Ensure the new region is within the bounds of the video_data
    new_roi_start_x = max(0, new_roi_start_x)
    new_roi_start_y = max(0, new_roi_start_y)
    new_roi_end_x = min(avg_flash.shape[2], new_roi_start_x + new_roi_width)
    new_roi_end_y = min(avg_flash.shape[1], new_roi_start_y + new_roi_height)

    # Extract the data within the new region for the specific time frame
    new_region = avg_flash[:, new_roi_start_y:new_roi_end_y, new_roi_start_x:new_roi_end_x]

    return new_region

def animate_figure(avg_flash, fps=10, save_path="C:/Users/trapped/Documents/GitHub/corticalanalysis/Notebook/animated_graphs/animated_graph.gif"):
    interval_time = 1/fps

    x=np.arange(len(avg_flash))
    y=avg_flash.mean(axis=(1,2))
    fig, ax = plt.subplots()
    line, =ax.plot(x,y)
    def myupdating(i):
        line.set_data(x[:i],y[:i]) #x[:i] is the sub array of array x from position 0 to i-1

    myanimation = FuncAnimation(fig,myupdating, frames=len(avg_flash),interval=interval_time)
    myanimation.save(save_path, writer='pillow')

    print(f"GIF stack saved at: {save_path}")

def extract_v1_trials(avg_flash, roi_time = 5, roi_start_x=15, roi_start_y = 85, roi_width=45, roi_height = 45):
    
    #Determine the coordinates of the first region (ROI)
    first_region = avg_flash[:, roi_time, roi_start_y:roi_start_y+roi_height, roi_start_x:roi_start_x+roi_width]
    
    #Find the coordinates of the maximum value within the extracted region
    max_coords = np.unravel_index(np.argmax(first_region), first_region.shape)
    max_x, max_y = max_coords[1] + roi_start_x, max_coords[0] + roi_start_y

    new_roi_width, new_roi_height = 10, 10  # Adjust the dimensions of the new region
    new_roi_start_x = max_x - new_roi_width // 2
    new_roi_start_y = max_y - new_roi_height // 2
    
    # Ensure the new region is within the bounds of the video_data
    new_roi_start_x = max(0, new_roi_start_x)
    new_roi_start_y = max(0, new_roi_start_y)
    new_roi_end_x = min(avg_flash.shape[2], new_roi_start_x + new_roi_width)
    new_roi_end_y = min(avg_flash.shape[1], new_roi_start_y + new_roi_height)

    # Extract the data within the new region for the specific time frame
    new_region = avg_flash[:, :, new_roi_start_y:new_roi_end_y, new_roi_start_x:new_roi_end_x]

    return new_region

def trials(flash_indices, smoothed_signal, double = 'FALSE', segment_size = 330, minus=0):
    segments = []
    
    if double == 'TRUE':
        for flash_index in flash_indices:
            d_start_index = flash_index[0]- minus
            d_end_index = d_start_index + segment_size
            segments.append(smoothed_signal[d_start_index:d_end_index])
        
    else: 
        for flash_index in flash_indices:
            start_index = flash_index- minus
            end_index = flash_index + segment_size
            segments.append(smoothed_signal[start_index:end_index])
            
    segments_array = np.array(segments)
    return segments_array

def heatmap(array, x="frames (30fps)", y="trials", title_name = "single flash full frame trials"):

    sns.set_theme()

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap((array.mean(axis=(2,3))), cmap = "jet")
    #ax.set_yticklabels(len(full_single_trials.mean(axis=(2,3)) + 1))
    ax.set(xlabel=x, ylabel=y, title = title_name)
    #export_path = "X:/Raymond Lab/1_Kai/3_processed_data/4_heatmaps/single_stage3.png"
    #plt.savefig(export_path)

def extract_v1_trials_wo_spouts(avg_flash, roi_time = 5, roi_start_x=15, roi_start_y = 85, roi_width=45, roi_height = 45):
    
    #Determine the coordinates of the first region (ROI)
    first_region = avg_flash[:, roi_time, roi_start_y:roi_start_y+roi_height, roi_start_x:roi_start_x+roi_width]
    
    #Find the coordinates of the maximum value within the extracted region
    max_coords = np.unravel_index(np.argmax(first_region), first_region.shape)
    max_x, max_y = max_coords[1] + roi_start_x, max_coords[0] + roi_start_y

    new_roi_width, new_roi_height = 10, 10  # Adjust the dimensions of the new region
    new_roi_start_x = max_x - new_roi_width // 2
    new_roi_start_y = max_y - new_roi_height // 2
    
    # Ensure the new region is within the bounds of the video_data
    new_roi_start_x = max(0, new_roi_start_x)
    new_roi_start_y = max(0, new_roi_start_y)
    new_roi_end_x = min(avg_flash.shape[3], new_roi_start_x + new_roi_width)
    new_roi_end_y = min(avg_flash.shape[2], new_roi_start_y + new_roi_height)

    # Extract the data within the new region for the specific time frame
    new_region = avg_flash[:, :, new_roi_start_y:new_roi_end_y, new_roi_start_x:new_roi_end_x]

    return new_region

# brain registration
def calculate_and_adjust_coordinates(cortex_data, max_projection):
    all_x = []
    all_y = []
    # Gather all coordinates from the dataset to find extents
    for index, row in cortex_data.iterrows():
        all_x.extend(eval(row['left_x']) + eval(row['right_x']))
        all_y.extend(eval(row['left_y']) + eval(row['right_y']))
    
    # Calculate min and max
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    width = max_x - min_x
    height = max_y - min_y

    # Determine scaling factor to fit the overlay in the image dimensions
    scale_factor = min(max_projection.shape[0] / height, max_projection.shape[1] / width)

    # Calculate translation to center the overlay
    center_x = (max_projection.shape[1] - (width * scale_factor)) / 2
    center_y = (max_projection.shape[0] - (height * scale_factor)) / 2
    
    return scale_factor, center_x - min_x * scale_factor, center_y - min_y * scale_factor

# Adjust draw_static_boundaries function to include scaling and translation
def draw_adjusted_static_boundaries(scale, tx, ty, max_projection, cortex_data):
    overlay = np.zeros_like(max_projection, dtype=np.uint8)
    cortex_data['adjusted_left_x'] = None
    cortex_data['adjusted_left_y'] = None
    cortex_data['adjusted_right_x'] = None
    cortex_data['adjusted_right_y'] = None
    for index, row in cortex_data.iterrows():
        left_x = [x * scale + tx for x in eval(row['left_x'])]
        left_y = [y * scale + ty for y in eval(row['left_y'])]
        right_x = [x * scale + tx for x in eval(row['right_x'])]
        right_y = [y * scale + ty for y in eval(row['right_y'])]
        # Update the DataFrame with the adjusted coordinates
        cortex_data.at[index, 'adjusted_left_x'] = left_x
        cortex_data.at[index, 'adjusted_left_y'] = left_y
        cortex_data.at[index, 'adjusted_right_x'] = right_x
        cortex_data.at[index, 'adjusted_right_y'] = right_y
        
        left_coords = np.array(list(zip(left_x, left_y)), dtype=np.int32)
        right_coords = np.array(list(zip(right_x, right_y)), dtype=np.int32)
        
        cv2.polylines(overlay, [left_coords], isClosed=False, color=(255, 255, 0), thickness=1)
        cv2.polylines(overlay, [right_coords], isClosed=False, color=(255, 255, 0), thickness=1)
    
    return overlay

# Update the composite image function to use the adjusted overlay
def transform_image_only_adjusted(x_offset, y_offset, scale,max_projection):
    scale_factor_img = max(scale / 100, 0.01)
    transformed_img = cv2.resize(max_projection.astype(np.float32), None, fx=scale_factor_img, fy=scale_factor_img, interpolation=cv2.INTER_LINEAR)
    transformed_img = cv2.normalize(transformed_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    composite_image = np.zeros_like(max_projection, dtype=np.uint8)
    start_x = int(composite_image.shape[1]/2 - transformed_img.shape[1]/2 + x_offset)
    start_y = int(composite_image.shape[0]/2 - transformed_img.shape[0]/2 + y_offset)
    end_x = start_x + transformed_img.shape[1]
    end_y = start_y + transformed_img.shape[0]
    if start_x < 0 or start_y < 0 or end_x > composite_image.shape[1] or end_y > composite_image.shape[0]:
        sx = max(start_x, 0)
        sy = max(start_y, 0)
        ex = min(end_x, composite_image.shape[1])
        ey = min(end_y, composite_image.shape[0])
        composite_image[sy:ey, sx:ex] = transformed_img[sy-start_y:ey-start_y, sx-start_x:ex-start_x]
    else:
        composite_image[start_y:end_y, start_x:end_x] = transformed_img
    
    mask = boundary_overlay_adjusted > 0
    composite_image[mask] = boundary_overlay_adjusted[mask]
    
    return composite_image

# Update the display function to use the new composite function
def update_image_only_adjusted(val, max_projection):
    global x_offset, y_offset, scale
    x_offset = cv2.getTrackbarPos('X Offset', 'Registration') - 170
    y_offset = cv2.getTrackbarPos('Y Offset', 'Registration') - 170
    scale = cv2.getTrackbarPos('Scale', 'Registration')
    transformed = transform_image_only_adjusted(x_offset, y_offset, scale, max_projection)
    cv2.imshow('Registration', transformed)

def adjust_tif_stack(stack, scale, x_offset, y_offset):
    # Calculate scaling factor
    scale_factor = max(scale / 100, 0.01)
    adjusted_stack = []
    for frame in stack:
        # Scale each frame
        transformed_frame = cv2.resize(frame.astype(np.uint16), None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        # Create new blank frame to hold transformed frame
        new_frame = np.zeros_like(frame, dtype=np.uint16)
        start_x = int(new_frame.shape[1]/2 - transformed_frame.shape[1]/2 + x_offset)
        start_y = int(new_frame.shape[0]/2 - transformed_frame.shape[0]/2 + y_offset)
        end_x = start_x + transformed_frame.shape[1]
        end_y = start_y + transformed_frame.shape[0]
        if start_x < 0 or start_y < 0 or end_x > new_frame.shape[1] or end_y > new_frame.shape[0]:
            sx = max(start_x, 0)
            sy = max(start_y, 0)
            ex = min(end_x, new_frame.shape[1])
            ey = min(end_y, new_frame.shape[0])
            new_frame[sy:ey, sx:ex] = transformed_frame[sy-start_y:ey-start_y, sx-start_x:ex-start_x]
        else:
            new_frame[start_y:end_y, start_x:end_x] = transformed_frame
        adjusted_stack.append(new_frame)
    return np.array(adjusted_stack).astype(np.uint16)
    
def display_image_with_overlay(image, cortex_data, region_name, side = 'both'):
    # Filter the row corresponding to the region of interest
    region_row = cortex_data[cortex_data['acronym'] == region_name]
    
    if region_row.empty:
        print(f"No region found with the name {region_name}")
        return
    
    # Extract the coordinates from the filtered row
    left_x = region_row.iloc[0]['adjusted_left_x']
    left_y = region_row.iloc[0]['adjusted_left_y']
    right_x = region_row.iloc[0]['adjusted_right_x']
    right_y = region_row.iloc[0]['adjusted_right_y']
    
    # Convert coordinates to the correct format for plotting
    
    left_coords = np.array(list(zip(left_x, left_y)))
    right_coords = np.array(list(zip(right_x, right_y)))
    
    # Display the image
    plt.imshow(image, cmap='gray')
    
    # Plot the polylines on the image
    if side == 'both':
        plt.plot(left_coords[:, 0], left_coords[:, 1], color='yellow', linewidth=1)
        plt.plot(right_coords[:, 0], right_coords[:, 1], color='yellow', linewidth=1)

    if side == 'left':
        plt.plot(left_coords[:, 0], left_coords[:, 1], color='yellow', linewidth=1)

    if side == 'right':
        plt.plot(right_coords[:, 0], right_coords[:, 1], color='yellow', linewidth=1)
    
    # Show the image with overlay
    plt.title("Image with Overlay")
    plt.axis('off')
    plt.show()