# Processing of Calcium Mesoscale Cortical Activity into dff tiff stacks

## Dataset used: 
The dataset used in this project was videos of calcium dynamics in layer 2/3 of the cortex(stored as tiff stacks as approximately 70 000 depth x 128 height x 128 width). The videos were collected as the transgenic WT: or Q175:Thy1- GCaMP6s mouse was completing a task where they must lick the correct spout in response to a flash of light. These flashes of light create an artifact visible in the video. Thereby that artifact was identified and interpolated over, to allow for smooth analysis of the cortical data.

## Implementation

This project was completed in Python version 3.12 using the following packages and versions:

numpy 1.26.2
matplotlib 3.8.2
matplotlib inline 0.1.6
tifffile 2023.9.26
opencv-python 4.8.1.78
scipy 1.11.4
tqdm 4.66.1
plotly 5.9.0

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
