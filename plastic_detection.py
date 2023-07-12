from sklearnex import patch_sklearn
patch_sklearn()
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

rescale = 3


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


def display(img):
    """
    Display an image.

    Parameters
    ----------
    img : numpy.ndarray
        Image array to display.

    """
    # Show the image
    cv2.imshow('Image', img)
    
    # Wait for a key press
    cv2.waitKey(0)
    
    # Close the window
    cv2.destroyAllWindows()
    

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


def pixel_values(mask, hsi, rgb, x, y):
    """
    Get the pixel values for a given position in the mask, HSI, and RGB images.

    Parameters
    ----------
    mask : numpy.ndarray
        Mask image array.
    hsi : numpy.ndarray
        HSI image array.
    rgb : numpy.ndarray
        RGB image array.
    x : int
        x-coordinate of the pixel.
    y : int
        y-coordinate of the pixel.

    Returns
    -------
    int
        Label value.
    numpy.ndarray
        HSI pixel vector.
    numpy.ndarray
        RGB pixel vector.

    """
    label = mask[x, y]
    hsi_vector = hsi[x, y]
    rgb_vector = rgb[x, y]
    return label, hsi_vector, rgb_vector

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

def plot_stats_per_label_id(mask, rgb, hsi, mask_path):
    """
    Plot the mean pixel values per label ID.

    Parameters
    ----------
    mask : numpy.ndarray
        Mask image array.
    rgb : numpy.ndarray
        RGB image array.
    hsi : numpy.ndarray
        HSI image array.
    mask_path : str
        Path of the mask image.

    """
    fig, (ax) = plt.subplots( figsize = (6,4))
    for label_id in np.unique(mask):
        label_specific_mask = mask_for_each_label(mask, label_id)
        rgb_means, hsi_means = mean_pixel_values_for_label_id(mask, label_specific_mask, rgb, hsi)
        ax.plot(wavelengths, 
                hsi_means,
                linewidth=3,
                label= str(label_id),
                alpha = 1.0,
                marker="*")
    plt.xticks((np.arange((wavelengths[0] // 100)*100 , (wavelengths[-1] // 100 + 1)*100, 100)))
    ax.set_ylabel("Mean pixel value for the label")
    ax.set_xlabel('Wavelength')
    fig.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.30), markerscale=2.)
    plt.title(mask_path.replace("masks", "")[2:-15])
    plt.tight_layout()
    plt.draw()
    plt.savefig(str("hsi_stats/" + mask_path.replace("masks", "").replace("\\","_")[2:]), bbox_inches='tight', dpi=200)

def get_number_of_pixels_in_partition(path_list):
    """
    Get the number of pixels in a partition.
 
    Parameters:
    path_list (list): List of image paths.
 
    Returns:
    int: Number of pixels in the partition.
    """
    dataset_size = sum([cv2.imread(image_path).shape[0] * cv2.imread(image_path).shape[1] for image_path in path_list])#number_of_pixels
    return dataset_size 

def create_dataset(df, num_classes):
    """
    Create and return the dataset dictionary based on the given dataframe.
    
    Args:
        df (DataFrame): Dataframe containing the dataset information.
    
    Returns:
        dataset (dict): Dictionary containing the dataset splits and corresponding data.
    """
    # Check if the dataset file exists
    if os.path.isfile('dataset_scale_{}_classes_{}.pkl'.format(str(rescale), str(num_classes))):
        with open('dataset_scale_{}_classes_{}.pkl'.format(str(rescale), str(num_classes)), 'rb') as f:
            dataset = pickle.load(f)
        with open('scalers_scale_{}_classes_{}.pkl'.format(str(rescale), str(num_classes)), 'rb') as f:
            scalers = pickle.load(f)
    else:
        # Uncomment the following lines to print unique labels for each image
        # unique_labels = np.unique(np.concatenate(([np.unique(cv2.imread(img)) for img in train_mask_paths])))
        
        # Create dataset dictionary structure
        dataset = {"train": {"hsi": {},
                             "rgb": {},
                             "label": {}},
                   "test": {"hsi": {},
                            "rgb": {},
                            "label": {}},
                   "val": {"hsi": {},
                           "rgb": {},
                           "label": {}},
                   "finetune_train": {"hsi": {},
                                      "rgb": {},
                                      "label": {}},
                   "finetune_test": {"hsi": {},
                                     "rgb": {},
                                     "label": {}}}
        
        # Calculate train dataset size
        train_dataset_size = get_number_of_pixels_in_partition(df["path"][df["partition"]=="train"])#number_of_pixels
        # Initialize dataset splits
        hsi_dataset_split = np.zeros((train_dataset_size, load_hsi(find_pairs(df["path"][0])[0], rescale).shape[2]))
        rgb_dataset_split = np.zeros((train_dataset_size, load_rgb(find_pairs(df["path"][0])[1], rescale).shape[2]))
        labels_split = np.zeros((train_dataset_size))
        
        # Iterate over dataset splits
        for dataset_split in ["train", "test"]:
            index = 0
            # Iterate over rows in the current split
            for _, row in df[df["partition"]==dataset_split].iterrows():
                hsi_image_path, rgb_image_path = find_pairs(row["path"])
                hsi = load_hsi(hsi_image_path, rescale)
                rgb = load_rgb(rgb_image_path, rescale)
                mask = load_mask(row["path"], rescale)
                x = min(mask.shape[0], rgb.shape[0], hsi.shape[0])
                y = min(mask.shape[1], rgb.shape[1], hsi.shape[1])
                hsi = hsi[0:x, 0:y]
                rgb = rgb[0:x, 0:y]
                mask = mask[0:x, 0:y]
                
                n1, n2 = mask.shape
                number_of_valid_pixels = n1*n2
                hsi = hsi.reshape(n1*n2, -1)
                rgb = rgb.reshape(n1*n2, -1)
                labels = mask.reshape(n1*n2, -1)[:,0]# == unique_labels 
                
                # Exclude background if include_background is False
                if not row["include_background"]:
                    number_of_valid_pixels = number_of_valid_pixels - np.sum(mask==0)
                    hsi = hsi[mask.reshape(n1*n2, -1)[:,0]>0]
                    rgb = rgb[mask.reshape(n1*n2, -1)[:,0]>0]
                    labels = labels[mask.reshape(n1*n2, -1)[:,0]>0]
                    
                hsi_dataset_split[index : index + number_of_valid_pixels, :] = hsi
                rgb_dataset_split[index : index + number_of_valid_pixels, :] = rgb
                labels_split[index : index + number_of_valid_pixels] = labels
                index = index + number_of_valid_pixels
            
            # Update dataset with the split data
            dataset[dataset_split]["hsi"] = hsi_dataset_split[0:index]
            dataset[dataset_split]["rgb"] = rgb_dataset_split[0:index]
            dataset[dataset_split]["label"] = labels_split[0:index]
        # Split the train dataset into train and validation sets
        hsi_dataset_train, hsi_dataset_val, \
            labels_train, labels_val = \
                train_test_split(dataset["train"]["hsi"], 
                                 dataset["train"]["label"], test_size=0.50, random_state=42)
        rgb_dataset_train, rgb_dataset_val, \
            labels_train, labels_val = \
                train_test_split(dataset["train"]["rgb"], 
                                 dataset["train"]["label"], test_size=0.50, random_state=42)
        
        # Split the test dataset into finetune test and finetune train. This will be used for finetuning after training
        hsi_dataset_finetune_train, hsi_dataset_finetune_test, \
            labels_finetune_train, labels_finetune_test = \
                train_test_split(dataset["test"]["hsi"], 
                                 dataset["test"]["label"], test_size=0.90, random_state=42)
        rgb_dataset_finetune_train, rgb_dataset_finetune_test, \
            labels_finetune_train, labels_finetune_test = \
                train_test_split(dataset["test"]["rgb"], 
                                 dataset["test"]["label"], test_size=0.90, random_state=42)
        
        
        dataset["train"]["hsi"] = hsi_dataset_train
        dataset["train"]["label"] = labels_train
        dataset["val"]["hsi"] = hsi_dataset_val
        dataset["val"]["label"] = labels_val
        dataset["train"]["rgb"] = rgb_dataset_train
        dataset["val"]["rgb"] = rgb_dataset_val
        
        dataset["finetune_train"]["rgb"] = rgb_dataset_finetune_train
        dataset["finetune_train"]["hsi"] = hsi_dataset_finetune_train
        dataset["finetune_train"]["label"] = labels_finetune_train
        dataset["finetune_test"]["rgb"] = rgb_dataset_finetune_test
        dataset["finetune_test"]["hsi"] = hsi_dataset_finetune_test
        dataset["finetune_test"]["label"] = labels_finetune_test
        
        
        scalers = {}
        
        scalers["hsi"] = preprocessing.StandardScaler().fit(dataset["train"]["hsi"])
        scalers["rgb" ]= preprocessing.StandardScaler().fit(dataset["train"]["rgb"])
        # Convert labels to 3 classes
        for dataset_split in dataset:
            dataset[dataset_split]["label"] = np.ceil(dataset[dataset_split]["label"]/254)
            if num_classes == 2: #only detect plastic, all other classes are label 0
                dataset[dataset_split]["label"] = dataset[dataset_split]["label"] % 2
            for image_type in ["hsi", "rgb"]:
                dataset[dataset_split][image_type] = scalers[image_type].transform(dataset[dataset_split][image_type])
            
        # Save the dataset dictionary to a file
        with open('dataset_scale_{}_classes_{}.pkl'.format(str(rescale), str(num_classes)), 'wb') as f:
            pickle.dump(dataset, f)
        with open('scalers_scale_{}_classes_{}.pkl'.format(str(rescale), str(num_classes)), 'wb') as f:
            pickle.dump(scalers, f)

    return dataset, scalers

def predict_image(model_dict, image_mask_path, scalers, num_classes):
    """
    Generate and display predicted images based on the given models and image mask path.

    Args:
        model_dict (dict): Dictionary containing the trained models.
        image_mask_path (str): Path to the image mask.

    Returns:
        None
    """
    hsi_image_path, rgb_image_path = find_pairs(image_mask_path)
    hsi = load_hsi(hsi_image_path, rescale)
    rgb = load_rgb(rgb_image_path, rescale)
    mask = load_mask(image_mask_path, rescale)
    x = min(mask.shape[0], rgb.shape[0], hsi.shape[0])
    y = min(mask.shape[1], rgb.shape[1], hsi.shape[1])
    hsi = hsi[0:x, 0:y]
    rgb = rgb[0:x, 0:y]
    mask = mask[0:x, 0:y]
    
    if num_classes == 2:
        mask  = np.ceil(mask/254) % 2
    else:
        mask  = np.ceil(mask/254)
    titles = ["RGB Image",
              "True mask"]
    images = [rgb,
              mask]
    aucs = []
    reports = []
    # Predict and add model-specific images to the list
    for model_type in model_dict:
        print(model_type)
        titles.append("HSI " + model_type)
        titles.append("RGB " + model_type)
        # hsi_prediction = model_dict[model_type][0].predict(hsi.reshape(-1, hsi.shape[2])).reshape(hsi.shape[0:2])
        # rgb_prediction = model_dict[model_type][1].predict(rgb.reshape(-1, rgb.shape[2])).reshape(rgb.shape[0:2])
        hsi_prediction = model_dict[model_type][0].predict(scalers["hsi"].transform(hsi.reshape(-1, hsi.shape[2]))).reshape(hsi.shape[0:2])
        rgb_prediction = model_dict[model_type][1].predict(scalers["rgb"].transform(rgb.reshape(-1, rgb.shape[2]))).reshape(rgb.shape[0:2])
        if num_classes == 2:
            # print(np.amax(hsi_prediction))
            # print(mask, hsi_prediction)
            # print(model_dict[model_type][0].predict_proba(scalers["hsi"].transform(hsi.reshape(-1, hsi.shape[2]))).shape)
            # fpr, tpr, thresholds = metrics.roc_curve(np.ravel(mask), 
            #                                          model_dict[model_type][0].predict_proba(hsi.reshape(-1, hsi.shape[2]))[:,1])
            fpr, tpr, thresholds = metrics.roc_curve(np.ravel(mask), 
                                                      model_dict[model_type][0].predict_proba(scalers["hsi"].transform(hsi.reshape(-1, hsi.shape[2])))[:,1])
            hsi_auc = metrics.auc(fpr, tpr)
            fpr, tpr, thresholds = metrics.roc_curve(np.ravel(mask), 
                                                     model_dict[model_type][1].predict_proba(scalers["rgb"].transform(rgb.reshape(-1, rgb.shape[2])))[:,1])
            rgb_auc = metrics.auc(fpr, tpr)
            aucs.append(hsi_auc)
            aucs.append(rgb_auc)
            print(model_type, "hsi_auc, rgb_auc",hsi_auc, rgb_auc)
            # rgb_prediction = rgb_prediction > 0.5
            # hsi_prediction = hsi_prediction > 0.5
        images.append(hsi_prediction)
        images.append(rgb_prediction)
        report = metrics.classification_report(np.ravel(mask), np.ravel(hsi_prediction))
        print(report)
        # reports_hsi.append(report)
        # reports_rgb.append(report)
        print(np.max(mask), np.max(hsi_prediction))
    return images, titles, aucs



def inference_time(model_dict, image_mask_path, scalers, num_classes):
    """
    Generate and display predicted images based on the given models and image mask path.

    Args:
        model_dict (dict): Dictionary containing the trained models.
        image_mask_path (str): Path to the image mask.

    Returns:
        None
    """
    hsi_image_path, rgb_image_path = find_pairs(image_mask_path)
    hsi = load_hsi(hsi_image_path, rescale)
    rgb = load_rgb(rgb_image_path, rescale)
    mask = load_mask(image_mask_path, rescale)
    x = min(mask.shape[0], rgb.shape[0], hsi.shape[0])
    y = min(mask.shape[1], rgb.shape[1], hsi.shape[1])
    hsi = hsi[0:x, 0:y]
    rgb = rgb[0:x, 0:y]
    mask = mask[0:x, 0:y]
    
    if num_classes == 2:
        mask  = np.ceil(mask/254) % 2
    else:
        mask  = np.ceil(mask/254)
    titles = ["RGB Image",
              "True mask"]
    images = [rgb,
              mask]
    aucs = []
    reports = []
    import time
    # Predict and add model-specific images to the list
    for model_type in model_dict:
        print(model_type)
        titles.append("HSI " + model_type)
        titles.append("RGB " + model_type)
        input_ = scalers["hsi"].transform(hsi.reshape(-1, hsi.shape[2]))
        start_time = time.time()
        hsi_prediction = model_dict[model_type][0].predict(input_)
        end_time = time.time()
        print("hsi inference time",  end_time - start_time)
        start_time = time.time()
        input_ = scalers["rgb"].transform(rgb.reshape(-1, rgb.shape[2]))
        start_time = time.time()
        rgb_prediction = model_dict[model_type][1].predict(input_ )
        end_time = time.time()
        print("rgb inference time",  end_time - start_time)

def plot_predictions(images, titles, image_mask_path,  aucs):
    # Plot the images in separate figures
    fig, axes = plt.subplots(2,len(images)//2, figsize = (2*len(images),5))
    for i in range(len(images)):
        if i > 0:
            cmap = "gray"
            axes[i%2][i//2].imshow(images[i], cmap=cmap)
        else:
            axes[i%2][i//2].imshow(images[i]/np.amax(images[i]))
        if i > 1 and num_classes == 2:
            axes[i%2][i//2].imshow((images[1] == images[i]), cmap='prism', alpha=1)
            print("np.unique",np.unique(images[1] == images[i]))
        if i > 1 :
            score = balanced_accuracy_score(np.ravel(images[1]), np.ravel(images[i]))
            auc = aucs[i-2]
            axes[i%2][i//2].set_title(titles[i] + " ACC " +str(score)[0:4] + " AUC " + str(auc)[0:4])
        else:
            axes[i%2][i//2].set_title(titles[i])
        # Set the x and y-axis ticks to be invisible
        axes[i%2][i//2].set_xticks([])
        axes[i%2][i//2].set_yticks([])
    plt.suptitle(image_mask_path.replace("masks", "").replace("\\","_")[2:])
    plt.savefig(str("predictions/classes_" + str(num_classes) + image_mask_path.replace("masks", "").replace("\\","_").replace("//","_")[2:]), bbox_inches='tight', dpi=300)
    plt.show()
    
def plot_test_predictions(images_list, titles_list, aucs_list, df, mask_images):
    #%%
    fig, axes = plt.subplots(sum(df["partition"]== "test"),
                             len(titles_list[0]), figsize = (20,8))
    plot_index = 0
    # axes[i%2][i//2].set_title(titles[i])
    for index, model_name in enumerate(titles_list[0]):
        if index>1:
            axes[0][index].set_title(model_name[0:3], fontsize=18)
        else:
            axes[0][index].set_title(model_name, fontsize=18)
    for i in range(len(df["partition"])):
        axes[plot_index][0].set_ylabel("(" + ["a","b", "c", "d", "e", "f", "g", "h","i","j"][plot_index] + ")", rotation=0, fontsize=18)
        # axes[plot_index][0].set_ylabel(mask_images[i][2:-20].replace("_", "\n"), rotation=0)
        
        if (df["partition"]== "test")[i]: # if image is a test image
            axes[plot_index][0].imshow(images_list[i][0]/np.amax(images_list[i][0]))# plot the rgb image
            axes[plot_index][1].imshow(images_list[i][1],cmap="gray")# plot the mask image
            for k in range(2, len(titles_list[0])):
                axes[plot_index][k].imshow(images_list[i][k], cmap="gray")# plot the 
            # if and num_classes == 2:
                # axes[i%2][i//2].imshow((images[1] == images[i]), cmap='prism', alpha=1)
                # print("np.unique",np.unique(images[1] == images[i]))
            # if i > 1 :
                # score = balanced_accuracy_score(np.ravel(images[1]), np.ravel(images[i]))
                # auc = aucs[i-2]
                # axes[i%2][i//2].set_title(titles[i] + " ACC " +str(score)[0:4] + " AUC " + str(auc)[0:4])
            # else:
                # axes[i%2][i//2].set_title(titles[i])
            # Set the x and y-axis ticks to be invisible
            plot_index += 1
    for x in range(sum(df["partition"]== "test")):
        for y in range(len(titles_list[0])):
            axes[x][y].set_xticks([])
            axes[x][y].set_yticks([])
    # axes[0][2].text(-5.5, -5.5, "This is some text",fontsize = 20)
    # axes[0][4].text(-5.5, -5.5, "This is some text",fontsize = 20)
    # axes[0][6].text(-5.5, -5.5, "This is some text",fontsize = 20)
    plt.gcf().text(0.40, 0.9, "SVM", fontsize=22)
    plt.gcf().text(0.4+0.2, 0.9, "NN", fontsize=22)
    plt.gcf().text(0.4+0.4, 0.9, "LR", fontsize=22)
    plt.subplots_adjust(wspace=0.05, hspace=-0.1)
    # plt.suptitle(image_mask_path.replace("masks", "").replace("\\","_")[2:])
    plt.savefig(str("predictions/test_images_" + str(num_classes) + ".png"), bbox_inches='tight', dpi=300)
    plt.show()
    #%%
    


def train_model(clf_hsi, clf_rgb, dataset):
    """
    Train the given classifiers using the provided dataset.

    Args:
        clf_hsi: Classifier for HSI data.
        clf_rgb: Classifier for RGB data.
        dataset (dict): Dictionary containing the dataset splits.

    Returns:
        clf_hsi, clf_rgb: Trained classifiers.
    """
    # Fit the classifiers on the training data
    clf_hsi = clf_hsi.fit(dataset["train"]["hsi"],
                          dataset["train"]["label"])
    clf_rgb = clf_rgb.fit(dataset["train"]["rgb"],
                          dataset["train"]["label"])
    
    return clf_hsi, clf_rgb

def generate_scores(model_dict, dataset):
    """
    Generate and print model statistics based on the given model dictionary and dataset.

    Args:
        model_dict (dict): Dictionary containing the trained models.
        dataset (dict): Dictionary containing the dataset splits.

    Returns:
        None
    """
    # Initialize an empty dataframe to store model statistics
    df = pd.DataFrame({'model_type': [], 
                       'split': [],
                       'image_type': [],
                       'score': [],
                       'auc': []})
    metrics_dict = {"train": {"hsi":[],
                              "rgb":[]},
                    "val": {"hsi":[],
                            "rgb":[]},
                    "test": {"hsi":[],
                             "rgb":[]}}
    for index, image_type in enumerate(["hsi", "rgb"]):
        fig, axes = plt.subplots(2, 3, figsize = (15,10))
        for split_index, split  in enumerate(["train", "val", "test"]):
            reports = []
            for model_type in model_dict:
                print(split, image_type, model_type)
                # Calculate balanced accuracy score for each model, split, and image type
                prediction = model_dict[model_type][index].predict(dataset[split][image_type])
                if "binary" in model_type:
                    prediction = prediction > 0.5
                cm = confusion_matrix(dataset[split]["label"], prediction)
                disp = ConfusionMatrixDisplay(cm)
                disp.plot(ax=axes[index][split_index])
                # im = axes[index][split_index].imshow(cm, cmap="Blues")
                axes[index][split_index].set_title(split + " " + image_type)
                # axes[index][split_index].set_xlabels("Predicted")
                # axes[index][split_index].set_ylabel("True")
                score = balanced_accuracy_score(dataset[split]["label"], prediction)
                fpr, tpr, thresholds = metrics.roc_curve(dataset[split]["label"], 
                                                         model_dict[model_type][index].predict_proba(dataset[split][image_type])[:,1])
                auc = metrics.auc(fpr, tpr)

                report = metrics.classification_report(dataset[split]["label"], prediction,output_dict =True)
                # acc = report.split("accuracy")[1].split(".")[1][0:2]
                # print(acc)
                df_row = pd.DataFrame({'model_type': [model_type], 
                                       'split': [split],
                                       'image_type': [image_type],
                                       'score': [score],
                                       "auc":[auc]})
                print(split, image_type, report)
                # Append the statistics to the dataframe
                df = pd.concat([df, df_row])
                reports.append(report)
            metrics_dict[split][image_type] = reports
            # plt.colorbar(im, ax=axes[index][split_index])
        plt.suptitle(model_type)
        plt.savefig("confusion_matrix/" + model_type + "classes_" + str(np.amax(dataset["train"]["label"] + 1))+".png" , bbox_inches='tight', dpi=200)
        plt.show()
    return df, metrics_dict

    
def display_wavelength(mask_images):
    """
    Display the wavelength statistics for the given mask images.
    
    Args:
        mask_images (list): List of mask image paths.
    
    Returns:
        None
    """
    for mask_path in mask_images:
        hsi_image_path, rgb_image_path = find_pairs(mask_path)
        hsi = load_hsi(hsi_image_path)
        rgb = load_rgb(rgb_image_path)
        mask = load_mask(mask_path)
        x = min(mask.shape[0], rgb.shape[0], hsi.shape[0])
        y = min(mask.shape[1], rgb.shape[1], hsi.shape[1])
        hsi = hsi[0:x, 0:y]
        rgb = rgb[0:x, 0:y]
        mask = mask[0:x, 0:y]
        
        plot_stats_per_label_id(mask, rgb, hsi, mask_path)

def init_model(model_type, num_classes, load_trained_model):
    """
    Initialize and optionally load pre-trained classifiers based on the given model_type.

    Args:
        model_type (str): Type of the model to initialize.
        load_trained_model (bool): Flag indicating whether to load pre-trained models if available.

    Returns:
        clf_hsi, clf_rgb: Initialized classifiers for HSI and RGB data, respectively.
    """
    if load_trained_model:
        # Check if trained models already exist
        if os.path.isfile('models/model_hsi_{}_classes_{}.pkl'.format(model_type, str(num_classes))) and \
            os.path.isfile('models/model_rgb_{}_classes_{}.pkl'.format(model_type, str(num_classes))):
            clf_hsi = load('models/model_hsi_{}_classes_{}.pkl'.format(model_type, str(num_classes)))
            clf_rgb = load('models/model_rgb_{}_classes_{}.pkl'.format(model_type, str(num_classes)))
        else:
            load_trained_model = False
    if not load_trained_model:
        # If trained models do not exist, initialize classifiers based on model_type
        if model_type == "LDA":
            clf_hsi = LinearDiscriminantAnalysis()
            clf_rgb = LinearDiscriminantAnalysis()
        # elif model_type == "SVMsigmoid":
        #     clf_hsi = svm.SVC(kernel='sigmoid', C=1.0, max_iter=100)
        #     clf_rgb = svm.SVC(kernel='sigmoid', C=1.0, max_iter=100)
        # elif model_type == "SVMrbf":
        #     clf_hsi = svm.SVC(kernel='rbf', C=1.0, max_iter=100, probability=True)#play with c
        #     clf_rgb = svm.SVC(kernel='rbf', C=1.0, max_iter=100, probability=True)
        # elif model_type == "SVMrbfC01":
        #     clf_hsi = svm.SVC(kernel='rbf', C=0.1, max_iter=100, probability=True)#play with c
        #     clf_rgb = svm.SVC(kernel='rbf', C=0.1, max_iter=100, probability=True)
        elif model_type == "SVMlinear":
            clf_hsi = svm.SVC(kernel='linear', C=1.0, max_iter=100, probability=True)
            clf_rgb = svm.SVC(kernel='linear', C=1.0, max_iter=100, probability=True)
        # elif model_type == "SVMpoly":
        #     clf_hsi = svm.SVC(kernel='poly', degree=4, max_iter=100)
        #     clf_rgb = svm.SVC(kernel='poly', degree=4, max_iter=100)
        elif model_type == "LR":
            clf_hsi = LogisticRegression(multi_class='multinomial', 
                                         solver='lbfgs', max_iter=10000)
            clf_rgb = LogisticRegression(multi_class='multinomial', 
                                         solver='lbfgs', max_iter=10000)
        # elif model_type == "LR_binary":
        #     clf_hsi = LogisticRegression(max_iter=10000)
        #     clf_rgb = LogisticRegression(max_iter=10000)
        # elif model_type == "NN1":
        #     clf_hsi = MLPClassifier(hidden_layer_sizes=(99,),
        #                             activation='relu', solver='adam', max_iter=1000)
        #     clf_rgb = MLPClassifier(hidden_layer_sizes=(9,),
        #                             activation='relu', solver='adam', max_iter=1000)
        # elif model_type == "NN2":
        #     clf_hsi = MLPClassifier(hidden_layer_sizes=(100,50), 
        #                             activation='relu', solver='adam', max_iter=1000)
        #     clf_rgb = MLPClassifier(hidden_layer_sizes=(9,6),
        #                             activation='relu', solver='adam', max_iter=1000)
        elif model_type == "NN3v3":
            clf_hsi = MLPClassifier(hidden_layer_sizes=(100,50,25), 
                                    activation='relu', solver='adam', max_iter=500)
            clf_rgb = MLPClassifier(hidden_layer_sizes=(100,50,25),
                                    activation='relu', solver='adam', max_iter=500)
        # elif model_type == "NN1_binary":
        #     clf_hsi = MLPClassifier(hidden_layer_sizes=(100,), 
        #                             activation='sigmoid', solver='adam', max_iter=1000)
        #     clf_rgb = MLPClassifier(hidden_layer_sizes=(50,),
        #                             activation='sigmoid', solver='adam', max_iter=1000)
        # elif model_type == "NN2_binary":
        #     clf_hsi = MLPRegressor(hidden_layer_sizes=(100,50), 
        #                             activation='sigmoid', solver='adam', max_iter=1000)
        #     clf_rgb = MLPRegressor(hidden_layer_sizes=(9,6),
        #                             activation='sigmoid', solver='adam', max_iter=1000)
        # elif model_type == "NN3_binary":
        #     clf_hsi = MLPRegressor(hidden_layer_sizes=(99,66,33), 
        #                             activation='sigmoid', solver='adam', max_iter=1000)
        #     clf_rgb = MLPRegressor(hidden_layer_sizes=(9,6,3),
        #                             activation='sigmoid', solver='adam', max_iter=1000)
        elif model_type == "KNN":
            clf_hsi = KNeighborsClassifier(algorithm = "brute", n_neighbors=3, max_iter=2)
            clf_rgb = KNeighborsClassifier(algorithm = "brute", n_neighbors=3, max_iter=2)
    # Dump the initialized classifiers to files for future use
    dump(clf_hsi, 'models/model_hsi_{}_classes_{}.pkl'.format(model_type, str(num_classes)))
    dump(clf_rgb, 'models/model_rgb_{}_classes_{}.pkl'.format(model_type, str(num_classes)))
    return clf_hsi, clf_rgb

def finetune(clf_hsi, clf_rgb, dataset):
    # Fit the classifiers on parts of the test data
    # not used since test results are good enough
    clf_hsi = clf_hsi.fit(dataset["finetune_train"]["hsi"],
                          dataset["finetune_train"]["label"])
    clf_rgb = clf_rgb.fit(dataset["finetune_train"]["rgb"],
                          dataset["finetune_train"]["label"])
    return clf_hsi, clf_rgb
# display_wavelength(mask_images)
import pandas as pd

import datetime
import os
now = datetime.datetime.now()
time = now.strftime('%m_%d_%H:%M:%S')
mask_paths = ['./plastics_teaser/masks/0_labeled.png',
              './plastics_dry/masks/0_labeled.png',
              './plants_dry/masks/0_labeled.png',
              './plastics_dry_pure_sand/masks/0_labeled.png',
              './plastics_dry_wild_sand/masks/0_labeled.png',
               # './plastics_wet_pure/masks/0_labeled.png',#comment out
              './plastics_wild_segmentable/masks/0_labeled.png',
            
              './plastics_wet_wild_with_vegetation_settled/masks/0_labeled.png',
              './plastics_wet_wild_with_vegetation_turbid/masks/0_labeled.png',
              './plastics_wet_wild_settled/masks/0_labeled.png',
              './plastics_wet_wild_turbid/masks/0_labeled.png']
df = pd.DataFrame({'path': mask_paths, 
                    'partition': ["train",
                                  "train",
                                  "train",
                                  "train",
                                  "train",
                                  # "train",
                                  "train",
                                 
                                  "test",
                                  "test",
                                  "test",
                                  "test"],
                   'include_background': [True,
                                          False,
                                          True,
                                          True,
                                          # False,
                                          False,
                                          False,
                                          True,
                                          True,
                                          True,
                                          True]})





# # Example usage
# for mask_path in mask_images:
#     hsi_image_path, rgb_image_path = find_pairs(mask_path)
#     hsi = load_hsi(hsi_image_path,1)
#     rgb = load_rgb(rgb_image_path,1)
#     mask = load_mask(mask_path,1)
    
#     label_id = np.unique(mask)[1]
#     label_specific_mask = mask_for_each_label(mask, label_id)
#     # mean_pixel_values_for_label_id(mask, label_specific_mask, rgb, hsi)
#     plot_stats_per_label_id(mask, rgb, hsi, mask_path)


num_classes = 2
dataset, scaler = create_dataset(df, num_classes)
# model_list = [ "NN1","NN2"]
# model_list = [ "NN3"] 
# model_list = [ "LR", "LDA", "SVMrbf"]
# model_list = [ "NN1","NN2", "NN3", "LR", "LDA", "SVMrbf"]
# model_list = [ "LR", "LDA", "SVMrbf", "NN3v3","NN2", "NN1", "SVMrbfC01", "SVMlinear"]#,"NN2_binary", "NN3_binary", "LR_binary"]
model_list = ["SVMlinear_7_12", "NN3v3_7_12", "LR_7_12"]#,"NN2_binary", "NN3_binary", "LR_binary"]
model_list = ["NN3v3", "SVMlinear", "LR"]#,"NN2_binary", "NN3_binary", "LR_binary"]
model_dict = {}

for model_type in model_list:
    # try:
    print("model_type", model_type)
    load_trained_model=True
    clf_hsi, clf_rgb = init_model(model_type, num_classes, load_trained_model)
    if not load_trained_model:
        clf_hsi, clf_rgb = train_model(clf_hsi, clf_rgb, dataset)
    dump(clf_hsi, 'models/model_hsi_{}_classes_{}.pkl'.format(model_type, str(num_classes)))
    dump(clf_rgb, 'models/model_rgb_{}_classes_{}.pkl'.format(model_type, str(num_classes)))
    model_dict[model_type] = (clf_hsi, clf_rgb)
    # except:
        # pass
#fine_tune_model

scores_df, metrics_dict = generate_scores(model_dict, dataset)
print(scores_df)
scores_df.to_csv('scores.csv', index=False)

images_list = []
titles_list = []
auc_list = []
# mask_images = mask_images[4:5]
for mask_path in mask_paths:
    images, titles, aucs = predict_image(model_dict, mask_path, scaler, num_classes)
    plot_predictions(images, titles, mask_path, aucs)
    images_list.append(images)
    titles_list.append(titles)
    auc_list.append(aucs)
    
for mask_path in mask_paths:
    inference_time(model_dict, mask_path, scaler, num_classes)
    break
    

#%%
for index, model in enumerate(model_list):
    for split in ["train", "val", "test"]:
        for image_type in ["hsi", "rgb"]:
            metric = metrics_dict[split][image_type][index]
            print(model, split, image_type,
                  "acc", "precision", "recall","f1-score"), 
            print(round(metric["accuracy"], 2), "&",
                  round(metric["weighted avg"]["precision"], 2), "&",
                  round(metric["weighted avg"]["recall"], 2), "&",
                  round(metric["weighted avg"]["f1-score"], 2))
            print()
plot_test_predictions(images_list, titles_list, auc_list, df, mask_paths)