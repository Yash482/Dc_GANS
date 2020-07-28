import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

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


