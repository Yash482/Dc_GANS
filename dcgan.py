import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import numpy as np
import matplotlib as plt

#get dateset
#create dataloader
def get_dataloader(batch_size, image_size, data_dir = 'train/'):
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor()])
    dataset = dset.ImageFolder(data_dir, transform = transform )
    dataloader = torch.utils.data.DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True)
    return dataloader

# Setting some hyperparameters
batch_size = 256 # We set the size of the batch.
image_size = 32 # We set the size of the generated images (64x64).

train_loader = get_dataloader(batch_size, image_size)

"""
# data visvualization
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

#obtain batch of training images
data_iter = iter(train_loader)
imgs, _ = data_iter.next() #labels not required

#plot the imgs with labels
fig = plt.figure(figsize = (20, 4))
plot_size = 20
for i in np.arange(plot_size):
    ax = fig.add_subplot(2, plot_size/2, i+1, xticks = [], yticks = [])
    imshow(imgs[i])
"""

#scale the img pixel btw -1 to 1
def scale(x, feature_range=(-1, 1)):
    #this function assumes that the img is already scaled from 0-1
    min, max = feature_range
    x = x*(max- min) + min
    return x

























