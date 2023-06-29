# Hyperspectral Macroplastic Detection

## Code organization

### Processing 
Contains repos to process raw data into reflectance calibrated and registered images

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