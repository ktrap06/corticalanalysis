# Processing of Calcium Mesoscale Cortical Activity into dff tiff stacks
## Introduction
Processing **whole-brain calcium signals** in a behaving mouse is required to study how learning occurs in the cortex. In the field, it is very common to generate a _delta f over f for calcium data_, to be able to normalize the signal and determine what signals are neuronal activity, thereby removing hemodynamic or movement artifacts. Additionally, for this project, I am using visual stimuli to elicit **network activity in HD and WT mice**, this light is green and thereby creates an artifact in the cortical signal that needs to be removed for proper analysis. This script accomplishes this, by thresholding at 3 SD above the mean. Furthermore, I would like to include spatiotemporal analysis in this project via TCA, spatiotemporal PCA, and seeded pixel correlation analysis. This is possible using dff tiff stacks. 

Furthermore, I will be characterizing sensory spread in HD and WT mice usingg these scripts, and its relevance to behavioral learning. See [Sepers et al. (2022)](url) for reference.

## Dataset used: 
The dataset used in this project was videos of calcium dynamics in layer 2/3 of the cortex(stored as tiff stacks as approximately 70 000 depth x 128 height x 128 width). The videos were collected as the transgenic WT: or Q175:Thy1- GCaMP6s mouse was completing a task where they must lick the correct spout in response to a flash of light. These flashes of light create an artifact visible in the video. Thereby that artifact was identified and interpolated over, to allow for smooth analysis of the cortical data.

## Implementation

#### main_processing.ipynb 
This is the only script you will need. It contains 7 steps in a jupyter notebook to generate a dff tif stack, as well as associated figures throughout the processing of the signal. **PLEASE DUPLICATIE EACH TIME YOU PROCESS NEW DATA AND PLACE INTO A NOTEBOOKS FOLDER** Go through the step, first by giving the path to the file, and setting a frame you would like to see as the example frame. At the end you will need to also add the path where you would like the tiff output to be generated. 

#### preprocessing_functions.py
This contains all the functions that are used in the main_processing script. It contains the following functions in a .py file 
- load_frames - Loads in tif stacks as a 3d array. Make sure the dtype is 12.
- interactive_plot - Makes plotly interactive plots inline
- plot_frame - plots frame of interest with colormap
- temporal_mean - calculate and plot raw timeseries
- remove_dark_frames - remove darkframe from start and end, then plot timeseries
- extract_artifacts - removes artifacts from 3d array
- interpolate - interpolated the 5 removed artifact frames and plots timeseries
- dff - calculates delta f over f using a 10s moving average
- smoothing - spatial smoothing (gaussian, 3 pixel, and temporal chebeshev 1 filer
- make_video - generate dff video of whole session

#### Script-use examples folder
This contains a **main_notebook_example**: this is a jupyter notebook whereby I processed a pilot dataset. This is how each trial should look upon being processed and kept in a notebooks folder (personal use). 

**preprocessing_with_examples** is a jupyter notebook containing both preprocessing functions and the main script functions. This is used to generate the main script and preprocessing_function script. Not for dataset us, but rather generating more components to the project.

## Dependencies

This project was completed in Python version 3.12 using the following packages and versions:

- numpy 1.26.2
- matplotlib 3.8.2
- matplotlib inline 0.1.6
- tifffile 2023.9.26
- opencv-python 4.8.1.78
- scipy 1.11.4
- tqdm 4.66.1
- plotly 5.9.0

### Licensing:

This project is operating under an MIT free-use licence. All derived projects must include this licence, and the developers are not held responsible. Additionally all experiments have been approved by the Canadian Council on Animal Care under the Lynn Raymond Labratory at The University of British Columbia, thereby the MIT license was chosen.

### Todo list: 
- Add txt file of timestamps
- Make videos of averaged cortical response to stim
- Dff tiff stack with colormap
- Dppend onto CSV file with outputs
- Spatiotemporal PCA
- TCA
- Seeded pixel correlation
