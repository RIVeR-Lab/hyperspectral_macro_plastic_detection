# Hyperspectral Macroplastic Detection

## Code organization

### Processing 
Contains repos to process raw data into reflectance calibrated and registered images

### Experiments

`plastics_dry` - Reference plastic tiles in black plastic container

`plastics_dry_pure_sand` - Reference plastic tiles in dry sand river bed

`plastics_dry_wild_sand` - Real world containers in dry sand river bed

`plastics_teaser` - Plastic tiles scattered in real sand environment

`plastics_wet_pure` - Plastic reference tiles submerged under an 5 cm of water (if tiles were less bouyant than water)

`plastics_wet_wild_settled` - Real world containers in wet river bed with settled water

`plastics_wet_wild_turbid` - Real world containers in wet river bed with turbid, cloudy water

### Calibration
Contains homographies and dark/white reference images to normalize images, and mask to extract usable overlap range.

## Loading data

### Cube

Hyperspectral data is stored as a compressed numpy object for efficiency.
```
import numpy as np

# This loads a dictionary
data = np.load(<<PATH to npz file>>)
# Extract the 3D datacube
cube = data['cube']
```
### RGB Image
```
import cv2

img = cv2.cvtColor(cv2.imread(<<PATH to png file>>), cv2.COLOR_BGR2RGB)
```
### Labels
```
import cv2

img = cv2.imread(<<PATH to png file>>), cv2.IMREAD_UNCHANGED)
```