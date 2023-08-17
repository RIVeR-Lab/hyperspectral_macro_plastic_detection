import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import skimage
import matplotlib.pyplot as plt

# training_mask_paths = ['./plastics_teaser/masks/0_labeled.png']
training_mask_paths = ['./plastics_teaser/masks/0_labeled.png',
                       './plastics_dry/masks/0_labeled.png',
                       './plants_dry/masks/0_labeled.png',
                       './plastics_dry_pure_sand/masks/0_labeled.png',
                       './plastics_dry_wild_sand/masks/0_labeled.png',
                       # './plastics_wet_pure/masks/0_labeled.png',#comment out
                       './plastics_wild_segmentable/masks/0_labeled.png']
# test_mask_paths = ['./plastics_wet_wild_with_vegetation_settled/masks/0_labeled.png',
#                   './plastics_wet_wild_turbid/masks/0_labeled.png']
test_mask_paths = ['./plastics_wet_wild_with_vegetation_settled/masks/0_labeled.png',
                   './plastics_wet_wild_with_vegetation_turbid/masks/0_labeled.png',
                   './plastics_wet_wild_settled/masks/0_labeled.png',
                   './plastics_wet_wild_turbid/masks/0_labeled.png']

unique_labels = np.unique(np.concatenate(
    ([np.unique(cv2.imread(img)) for img in training_mask_paths])))
num_class = 2

rescale = 4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256,256))
    
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    
])

class tool():
    def __init__(self)->None:
        return
    
    def find_pairs(self, mask_path):
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
        hsi_image_path = mask_path.replace(
            "masks", "hsi_cal_registered").replace("labeled.png", "cube.npz")
        rgb_image_path = mask_path.replace(
            "masks", "rgb_registered").replace("labeled.png", "rgb.png")
        return hsi_image_path, rgb_image_path
    
    def load_hsi(self, path, rescale):
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
        cube = np.nan_to_num(cube, nan=0.0001, posinf=1)
        cube = skimage.measure.block_reduce(cube, (rescale, rescale, 1), np.mean)
        return cube
    
    def load_mask(self, path, rescale):
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
        if len(img.shape) == 3:
            img = img [:,:,0]

        img = skimage.measure.block_reduce(img, (rescale, rescale), np.max)
        return img
    
    def load_rgb(self, path, rescale):
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
        img = skimage.measure.block_reduce(img, (rescale, rescale, 1), np.mean)
        return img

class dataset(Dataset):
    def __init__(self, train_paths, is_scale:bool, transform):
        self.train_paths_ = train_paths
        self.t = tool()
        self._rescale = 1
        if (is_scale):
            self._rescale = rescale
        self.transform = transform

    def __len__(self) -> int:
        return len(self.train_paths_)
    
    def __getitem__(self, index):
        # get train image and mask
        hsi_path, _ = self.t.find_pairs(self.train_paths_[index])
        mask_img = self.t.load_mask(self.train_paths_[index], self._rescale)
        hsi_img = self.t.load_hsi(hsi_path, self._rescale)
        x = min(mask_img.shape[0], hsi_img.shape[0])
        y = min(mask_img.shape[1], hsi_img.shape[1])
        hsi = hsi_img[0:x, 0:y]
        mask = mask_img[0:x, 0:y]
        mask = self._get_labels(mask, num_class)
        hsi = self.transform(hsi)
        mask = self.transform(mask)
        return torch.tensor(hsi).to(torch.float), torch.tensor(mask).to(torch.long)

    def _get_labels(self, mask, num_class):
        # labels = np.zeros([mask.shape[0], mask.shape[1], num_class])
        labels = np.zeros([mask.shape[0], mask.shape[1]])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if mask[i,j]!=0:
                    labels[i,j] = 1
                else:
                    labels[i,j] = 0
        return labels
    
class cnn_hsi(nn.Module):
    def __init__(self, classes) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(33, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.ConvTranspose2d(64, classes, kernel_size=2, stride = 2)  
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, input):
        input = torch.relu(self.conv1(input))
        input = torch.relu(self.conv2(input))
        input = self.maxpool(input)
        input = torch.relu(self.conv3(input))
        input = torch.relu(self.conv3(input))
        input = self.conv4(input)
        return input


def train():
    num_epoch = 30
    #cnn_model = cnn(len(unique_labels)) #!!!
    cnn_model = cnn_hsi(2)
    optimizer = optim.Adam(cnn_model.parameters(), lr=0.007)
    criterion = nn.CrossEntropyLoss()
    # datasets
    train_dataset = dataset(training_mask_paths, True, transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    softmax = nn.Softmax(dim=1)
    for epoch in range(num_epoch):
        cnn_model.train()
        train_loss = 0
        for hsi, mask in train_loader:
            optimizer.zero_grad()
            output = cnn_model(hsi)
            num_class = output.shape[1]
            output = softmax(output)
            loss = criterion(output.permute(0,2,3,1).contiguous().view(-1,num_class), 
                             mask.permute(0,2,3,1).contiguous().view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        print("epoch:(%d/%d) ---"%(epoch+1,num_epoch),"train_loss:",train_loss)
        # print("epoch:(%d/%d) ---"%(epoch+1,num_epoch),"train_loss:",train_loss\
        #       ,"learning rate:",optimizer.param_groups[0]['lr'])
    return cnn_model

def test_one_image(cnn_model, image_paths:list):
    test_dataset = dataset(image_paths, True, transform=transform_test)
    test_loader = DataLoader(test_dataset,batch_size=1)
    cnn_model.eval()
    all_predictions = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for image, _ in test_loader:
            output = cnn_model(image)
            output = softmax(output)
            all_predictions.append(output)
            print(output.shape)
    colors = np.linspace(0,255,2)
    result = np.zeros((all_predictions[0].shape[2], all_predictions[0].shape[3]))
    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            values = all_predictions[0][0,:,i,j]
            # if max(prob) > 0.95:
            #     result[i][j] = 250
            # else:
            #     result[i][j] = 0
            result[i][j] = colors[np.argmax(values)]
    return result

def test(cnn_model, output_path):
    t = tool()
    plt.figure(1)
    n_row = 3
    n_col = len(training_mask_paths) + len(test_mask_paths)
    # first row, true mask img
    for i in range(n_col):
        if i>=len(training_mask_paths):
            test_i = i - len(training_mask_paths)
            _, rgb_path = t.find_pairs(test_mask_paths[test_i])
            rgb_img = t.load_rgb(rgb_path,rescale)
        else:
            _, rgb_path = t.find_pairs(training_mask_paths[i])
            rgb_img = t.load_rgb(rgb_path,rescale)
        plt.subplot(n_row,n_col,0*n_col+1+i)
        plt.imshow(rgb_img/255)
        plt.axis('off')
    # second row, true mask img
    for i in range(n_col):
        if i>=len(training_mask_paths):
            test_i = i - len(training_mask_paths)
            mask_img = t.load_mask(test_mask_paths[test_i],rescale)
        else:
            mask_img = t.load_mask(training_mask_paths[i],rescale)
        plt.subplot(n_row,n_col,1*n_col+1+i)
        plt.imshow(mask_img,'gray')
        plt.axis('off')
    # third row, model results
    for i in range(n_col):
        if i>=len(training_mask_paths):
            test_i = i - len(training_mask_paths)
            result = test_one_image(cnn_model, [test_mask_paths[test_i]])
        else:
            result = test_one_image(cnn_model, [training_mask_paths[i]])
        plt.subplot(n_row,n_col,2*n_col+1+i)
        plt.imshow(result,'gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(output_path,dpi=300)    
    plt.show()
    return

cnn_model = train()
torch.save(cnn_model,"../cnn_hsi_model1.pth")

model_path = "../cnn_models/cnn_hsi_model1.pth"
cnn_model = torch.load(model_path)
print(">load model finished.")
test(cnn_model, "../imgs/1.png")
