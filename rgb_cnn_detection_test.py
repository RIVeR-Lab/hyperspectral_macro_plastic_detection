import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import skimage
import matplotlib.pyplot as plt
from torchvision.models.vgg import VGG

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
    transforms.Resize((262,396))
    
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
        _, rgb_path = self.t.find_pairs(self.train_paths_[index])
        mask_img = self.t.load_mask(self.train_paths_[index], self._rescale)
        rgb_img = self.t.load_rgb(rgb_path, self._rescale)
        x = min(mask_img.shape[0], rgb_img.shape[0])
        y = min(mask_img.shape[1], rgb_img.shape[1])
        rgb = rgb_img[0:x, 0:y]
        mask = mask_img[0:x, 0:y]
        #mask = self._get_labels(mask, num_class)
        rgb = self.transform(rgb)
        mask = self.transform(mask)
        mask = onehot(mask,2)
        return torch.tensor(rgb).to(torch.float), torch.tensor(mask).to(torch.long)

    def _get_labels(self, mask, num_class):
        # labels = np.zeros([mask.shape[0], mask.shape[1], num_class])
        labels = np.zeros([mask.shape[0], mask.shape[1]])
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if mask[i,j] == 0 or mask[i,j] == 2:
                    labels[i,j] = 0
                else:
                    labels[i,j] = 1
        return labels
    








# =============== FCN START ======================

fcn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

class rgb_dataset(Dataset):
    def __init__(self, paths, fcn_transform=None):
        self.transform = fcn_transform
        self.t = tool()
        self.paths_ = paths
    def __len__(self):
        return len(self.paths_)
    def __getitem__(self, index):
        # get train image and mask
        _, rgb_path = self.t.find_pairs(self.paths_[index])
        imgA = cv2.imread(rgb_path)
        imgA = cv2.resize(imgA, (480,480))
        imgB = cv2.imread(self.paths_[index]) 
        if len(imgB.shape) == 3:
            imgB = imgB[:,:,0]  
        imgB = cv2.resize(imgB, (480,480))
        # imgB = imgB/255
        # imgB = imgB.astype('uint8')  
        imgB = onehot(imgB,2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        if self.transform:
            imgA = self.transform(imgA) 

        return imgA, imgB

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}
class VGGNet(VGG):
    #self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False
    #self, pretrained, model, requires_grad, remove_fc, show_params
    def __init__(self, pretrained=False, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
        #self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

class FCNs(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1) 
        # classifier is 1x1 conv, to reduce channels from 32 to n_class

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  
        x4 = output['x4']  
        x3 = output['x3']  
        x2 = output['x2']  
        x1 = output['x1']  

        score = self.bn1(self.relu(self.deconv1(x5)))     
        score = score + x4                                
        score = self.bn2(self.relu(self.deconv2(score)))  
        score = score + x3                                
        score = self.bn3(self.relu(self.deconv3(score)))  
        score = score + x2                                
        score = self.bn4(self.relu(self.deconv4(score)))  
        score = score + x1                                
        score = self.bn5(self.relu(self.deconv5(score)))  
        score = self.classifier(score)                    

        return score  

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    # nmsk = np.arange(data.size)*n + data.ravel()
    # buf.ravel()[nmsk-1] = 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]>2:
                buf[i,j,1]=1
            else:
                buf[i,j,0]=1
    return buf


# ================  FCN END ==================

def fcn_train():
    #self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False
    show_vgg_params = False
    vgg_model = VGGNet(requires_grad=True, show_params=show_vgg_params)
    fcn_model = FCNs(pretrained_net=vgg_model, n_class=2)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(fcn_model.parameters(), lr=1e-2, momentum=0.7)
    train_dataset = rgb_dataset(training_mask_paths, fcn_transform)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)


    epoch_num = 50
    # start train
    for epo in range(epoch_num):
        train_loss = 0
        fcn_model.train()
        for index, (bag, bag_msk) in enumerate(train_loader):
            # bag.shape is torch.Size([4, 3, 160, 160])
            # bag_msk.shape is torch.Size([4, 2, 160, 160])
            optimizer.zero_grad()
            output = fcn_model(bag)
            output = torch.sigmoid(output) # output.shape is torch.Size([4, 2, 160, 160])
            loss = criterion(output, bag_msk)
            loss.backward()
            iter_loss = loss.item()
            train_loss += iter_loss
            optimizer.step()
        print("epoch:(%d/%d) ---"%(epo+1,epoch_num),"train_loss:",train_loss)
        # if (epo+1)%10 == 0:
        #     fcn_test(fcn_model,[training_mask_paths[0]])

    return fcn_model

def fcn_test(fcn_model, test_path):
    fcn_model.eval()
    test_dataset = rgb_dataset(test_path, fcn_transform)
    test_loader = DataLoader(test_dataset, batch_size=1)
    all_predictions = []
    with torch.no_grad():
        for bag, bag_msk in test_loader:
            output = fcn_model(bag)
            output = torch.sigmoid(output) 

            all_predictions.append(output)
    output = all_predictions[0]
    output_np = output.cpu().detach().numpy().copy()
    output_np = np.argmax(output_np, axis=1)
    # result_img = output_np[:,None,:,:]
    # plot_img = np.zeros((result_img.shape[2],result_img.shape[3]))
    # for i in range(plot_img.shape[0]):
    #     for j in range(plot_img.shape[1]):
    #         plot_img[i,j] = result_img[0,0,i,j]
    # plt.figure(1)
    # plt.imshow(output_np[0,:,:],'gray')
    # plt.show()
    return output_np[0,:,:]

def plot_results(fcn_model, output_path):
    paths = training_mask_paths+test_mask_paths
    t = tool()
    plt.figure(1)
    n_row = 3
    n_col = len(paths)
    # first row, true mask img
    for i in range(n_col):
        _, rgb_path = t.find_pairs(paths[i])
        rgb_img = t.load_rgb(rgb_path,rescale)
        plt.subplot(n_row,n_col,0*n_col+1+i)
        plt.imshow(rgb_img/255)
        plt.axis('off')
    # second row, true mask img
    for i in range(n_col):
        mask_img = t.load_mask(paths[i],rescale)
        plt.subplot(n_row,n_col,1*n_col+1+i)
        plt.imshow(mask_img,'gray')
        plt.axis('off')
    # third row, model results
    for i in range(n_col):
        img = fcn_test(fcn_model,[paths[i]])
        plt.subplot(n_row,n_col,2*n_col+1+i)
        plt.imshow(img,'gray')
        plt.axis('off')
    plt.subplots_adjust(hspace=0.05)
    plt.savefig(output_path,dpi=300)    
    plt.show()
    return

if __name__ == "__main__":
    fcn_model = fcn_train()
    torch.save(fcn_model,"../cnn_models/fcn_rgb_model2.pth")

    # fcn_model = torch.load("../cnn_models/fcn_rgb_model1.pth")
    # fcn_test(fcn_model,[training_mask_paths[0]])

    fcn_model = torch.load("../cnn_models/fcn_rgb_model2.pth")
    plot_results(fcn_model, "../imgs/fcn_rgb_2.png")
