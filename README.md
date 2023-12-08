# Processing of Calcium Mesoscale Cortical Activity into dff tiff stacks

## Dataset used: 
The dataset used in this project was videos of calcium dynamics in layer 2/3 of the cortex(stored as tiff stacks as approximately 70 000 depth x 128 height x 128 width). The videos were collected as the transgenic WT: or Q175:Thy1- GCaMP6s mouse was completing a task where they must lick the correct spout in response to a flash of light. These flashes of light create an artifact visible in the video. Thereby that artifact was identified and interpolated over, to allow for smooth analysis of the cortical data.

## Implementation

main_processing.ipynb is the only script you will need. It contains 7 steps in a jupyter notebook to generate a dff tif stack, as well as associated figures throughout the processing of the signal. **PLEASE DUPLICATIE EACH TIME YOU PROCESS NEW DATA AND PLACE INTO A NOTEBOOKS FOLDER** Go through the step, first by giving the path to the file, and setting a frame you would like to see as the example frame. At the end you will need to also add the path where you would like the tiff output to be generated. 

preprocessing_functions.py contains all the functions that are used in the main_processing script. 

The Script-use examples contains a

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
