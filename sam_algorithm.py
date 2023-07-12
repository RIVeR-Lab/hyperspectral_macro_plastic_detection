# from sklearnex import patch_sklearn
# patch_sklearn()
import cv2
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from sklearn.metrics import balanced_accuracy_score
import pickle 
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
import os
from joblib import dump, load
# from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import skimage.measure
import plot_utils
# Load images
# rgb_images = glob.glob("./*/rgb*/*")  # Get paths of RGB images
# hsi_images = glob.glob("./*/hsi*/*")  # Get paths of HSI images
# mask_images = glob.glob("./*/masks*/*")  # Get paths of mask images
wavelengths = np.load("wavelengths.npy")[0]

# rescale = 1


def load_rgb(path, rescale):
    """
    Load and return an RGB image.

    Parameters
    ----------
    path : str
        Path of the image file.

    Returns
    -------
    numpy.ndarray
        RGB image array.

    """
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = skimage.measure.block_reduce(img, (rescale,rescale,1), np.mean)
    return img

def load_mask(path, rescale):
    """
    Load and return a mask image.

    Parameters
    ----------
    path : str
        Path of the image file.

    Returns
    -------
    numpy.ndarray
        Mask image array.

    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = img [:,:,0]
    img = skimage.measure.block_reduce(img, (rescale,rescale), np.max)
    return img 

def load_hsi(path, rescale):
    """
    Load and return an HSI image.

    Parameters
    ----------
    path : str
        Path of the image file.

    Returns
    -------
    numpy.ndarray
        HSI image array.

    """
    # This loads a dictionary
    data = np.load(path)
    # Extract the 3D datacube
    cube = data['cube']
    cube = np.nan_to_num(cube,nan=0.0001, posinf=1)
    cube = skimage.measure.block_reduce(cube, (rescale,rescale,1), np.mean)
    return cube

# normalize the grid values
def normalize(data):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = data.min()
    array_max = data.max()

    return ((data - array_min) / (array_max - array_min))
    # return new_array


def find_pairs(mask_path):
    """
    Find corresponding paths for HSI image and RGB image based on the mask image path.

    Parameters
    ----------
    mask_path : str
        Path of the mask image.

    Returns
    -------
    str
        Path of the HSI image.
    str
        Path of the RGB image.

    """
    hsi_image_path = mask_path.replace("masks", "hsi_cal_registered").replace("labeled.png", "cube.npz")
    rgb_image_path = mask_path.replace("masks", "rgb_registered").replace("labeled.png", "rgb.png")
    return hsi_image_path, rgb_image_path 

training_mask_paths = ['./plastics_teaser/masks/0_labeled.png',
              './plastics_dry/masks/0_labeled.png',
              './plants_dry/masks/0_labeled.png',
              './plastics_dry_pure_sand/masks/0_labeled.png',
              './plastics_dry_wild_sand/masks/0_labeled.png',
               # './plastics_wet_pure/masks/0_labeled.png',#comment out
              './plastics_wild_segmentable/masks/0_labeled.png']
test_mask_paths = ['./plastics_wet_wild_with_vegetation_settled/masks/0_labeled.png',
                  './plastics_wet_wild_with_vegetation_turbid/masks/0_labeled.png',
                  './plastics_wet_wild_settled/masks/0_labeled.png',
                  './plastics_wet_wild_turbid/masks/0_labeled.png']


def mean_pixel_values_for_label_id(mask, label_id_mask, rgb, hsi):
    """
    Calculate the mean pixel values for a specific label ID.

    Parameters
    ----------
    mask : numpy.ndarray
        Mask image array.
    label_id_mask : numpy.ndarray
        Binary mask for the label ID.
    rgb : numpy.ndarray
        RGB image array.
    hsi : numpy.ndarray
        HSI image array.

    Returns
    -------
    list
        Mean pixel values for the RGB channels.
    list
        Mean pixel values for the HSI channels.

    """
    rgb_means = [np.mean(np.multiply(rgb[:,:,channel], label_id_mask))/np.mean(label_id_mask) for channel in range(rgb.shape[2])]
    hsi_means = [np.mean(np.multiply( np.nan_to_num(hsi[:,:,channel]), label_id_mask))/np.mean(label_id_mask) for channel in range(hsi.shape[2])]

    return rgb_means, hsi_means


def mask_for_each_label(mask, label_id):
    """
    Create a binary mask for a specific label ID.

    Parameters
    ----------
    mask : numpy.ndarray
        Mask image array.
    label_id : int
        Label ID.

    Returns
    -------
    numpy.ndarray
        Binary mask for the label ID.

    """
    label_specific_mask = mask == label_id
    return label_specific_mask

rescale = 2
accs = []
unique_labels = np.unique(np.concatenate(([np.unique(cv2.imread(img)) for img in training_mask_paths])))
ref_vect = {}
for label in unique_labels:
    ref_vect[label] = []
# generate_dictionary
for plot_index, mask_path in enumerate(training_mask_paths):
    hsi_image_path, rgb_image_path = find_pairs(mask_path)
    hsi = load_hsi(hsi_image_path, rescale)
    rgb = load_rgb(rgb_image_path, rescale)
    mask = load_mask(mask_path, rescale)
    x = min(mask.shape[0], rgb.shape[0], hsi.shape[0])
    y = min(mask.shape[1], rgb.shape[1], hsi.shape[1])
    hsi = hsi[0:x, 0:y]
    rgb = rgb[0:x, 0:y]
    mask = mask[0:x, 0:y]
    is_plastic_mask = np.ceil(mask/254) % 2
    # break    
    
    ref_count = 0
    plt.figure(2)
    ref_cord = []

    ix = hsi.shape[0]
    iy = hsi.shape[1]
    bands = hsi.shape[2]
    
    # ref_vect is refernce pixel
    # ref_vect = [[0 for x in range(bands)] for y in range(ref_count)]
    
    # for i in range(ref_count):
    #     for j in range(bands):
    #         ref_vect[i][j] = arr[j, int(ref_cord[i][1]), int(ref_cord[i][0])]
    # ref_vect = []
    for index, label_id in enumerate(np.unique(mask)):
        label_specific_mask = mask_for_each_label(mask, label_id)
        rgb_means, hsi_means = mean_pixel_values_for_label_id(mask, label_specific_mask, rgb, hsi)
        ref_vect[label_id].append(hsi_means)
        # ref_vect[label_id].append(hsi_means)
    print(ref_vect)

#average spectra for each class
for label in ref_vect:
    ref_vect[label] = np.mean(np.array(ref_vect[label]),axis=0) 

ref_vect

fig, axes = plt.subplots(1,2, figsize = (10,5))

for label in ref_vect:
    axes[0].plot(wavelengths, 
            ref_vect[label],
            linewidth=3,
            label = label,
            alpha = 1.0,
            marker="*")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
fig.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.30), markerscale=2.)
    
    # acc = np.sum(is_plastic_mask == is_plastic_prediction)/(is_plastic_prediction.shape[0]*is_plastic_prediction.shape[1])
    # accs.append(acc)
    # print(mask_path, "Acc", acc)

            # score = balanced_accuracy_score(np.ravel(images[1]), np.ravel(images[i]))
            # axes[i%2][i//2].set_title(titles[i] + " ACC " +str(score)[0:4] + " AUC " + str(auc)[0:4])
    # plt.suptitle(image_mask_path.replace("masks", "").replace("\\","_")[2:])
plt.show()

def detect(mask_paths, ref_vect):
    
    #run classification
    fig, axes = plt.subplots(len(mask_paths), 5, figsize = (4,5))
    axes[0][0].set_title("Multiclass \n mask",fontsize=8)
    axes[0][1].set_title("Multiclass \n prediction",fontsize=8)
    axes[0][2].set_title("Plastic",fontsize=8)
    axes[0][3].set_title("Plastic \n prediction",fontsize=8)
    axes[0][4].set_title("True \n classifications",fontsize=8)
    class_color = {0: [255, 0, 0], 
                   1: [0, 255, 0], 
                   2: [0, 0, 255], 
                   3: [255, 0, 255], 
                   4: [0, 200, 200],
                   5: [200, 200, 0], 
                   6: [100, 100, 0], 
                   7: [250, 255, 80], 
                   8: [200, 0, 100], 
                   9: [100, 0, 100], 
                   10: [0, 0, 55],
                   11: [0, 55, 0],
                   12: [55, 0, 0],
                   12: [55, 55, 0]}
    for plot_index, mask_path in enumerate(mask_paths):
        hsi_image_path, rgb_image_path = find_pairs(mask_path)
        hsi = load_hsi(hsi_image_path, rescale)
        rgb = load_rgb(rgb_image_path, rescale)
        mask = load_mask(mask_path, rescale)
        x = min(mask.shape[0], rgb.shape[0], hsi.shape[0])
        y = min(mask.shape[1], rgb.shape[1], hsi.shape[1])
        hsi = hsi[0:x, 0:y]
        rgb = rgb[0:x, 0:y]
        mask = mask[0:x, 0:y]
        is_plastic_mask = np.ceil(mask/254) % 2
        ref_count = len(ref_vect)
    
        # default palettte made to classsify
        c = np.ndarray((ix, iy, 3))
        pred = np.ndarray((ix, iy))
        # angle_matrix = np.zeros((ix, iy, len(ref_vect), ))
        for x in range(ix):
            for y in range(iy):
                p_ang = [0] * ref_count
                for j, j_label in enumerate(ref_vect):
                    p_mag = [0] * ref_count
                    product_numtr = [0] * ref_count
                    p_deno = [0] * ref_count
                    p_r = [0] * ref_count
                    p_reno = [0] * ref_count
                    p_cos = [0] * ref_count
                    for nb in range(bands):
                        t = hsi[x, y, nb]
                        r = ref_vect[j_label][nb]
                        product_numtr[j] += (t * r)
                        p_r[j] += (r * r)
                        p_mag[j] += (t * t)
                    p_deno[j] = (np.sqrt(p_mag[j]))
                    p_reno[j] = (np.sqrt(p_r[j]))
                    p_cos[j] = product_numtr[j] / (p_deno[j] * p_reno[j])
                    p_ang[j] = np.arccos(p_cos[j])
        
                # if min(p_ang) < 0.2:
                class_no = np.argmin(p_ang, axis=0)
                # else:
                    # class_no = ref_count+1
                pred[x, y] =  class_no
                c[x, y] = class_color[class_no]
        is_plastic_prediction = np.ceil(pred/np.amax(pred)/0.99) % 2
    
        for x in range(5):
            axes[plot_index][x].set_xticks([])
            axes[plot_index][x].set_yticks([])
        axes[plot_index][0].imshow(mask)
        axes[plot_index][1].imshow(pred/np.amax(pred))
        axes[plot_index][2].imshow(is_plastic_mask)
        axes[plot_index][3].imshow(is_plastic_prediction)
        axes[plot_index][4].imshow(is_plastic_mask == is_plastic_prediction)
        acc = np.sum(is_plastic_mask == is_plastic_prediction)/(is_plastic_prediction.shape[0]*is_plastic_prediction.shape[1])
        accs.append(acc)
        print(mask_path, "Acc", acc)
    plt.subplots_adjust(wspace=0.05, hspace=-0.1)
        
                # score = balanced_accuracy_score(np.ravel(images[1]), np.ravel(images[i]))
                # axes[i%2][i//2].set_title(titles[i] + " ACC " +str(score)[0:4] + " AUC " + str(auc)[0:4])
        # plt.suptitle(image_mask_path.replace("masks", "").replace("\\","_")[2:])
    plt.show()
        # plt.savefig("sam_" + mask_path[2:-20] + ".png", bbox_inches='tight', dpi=300)


detect(training_mask_paths, ref_vect)
detect(test_mask_paths, ref_vect)
