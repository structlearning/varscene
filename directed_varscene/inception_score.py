import pickle
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import glob
from PIL import Image
import os
import argparse

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    def get_pred(x):
        if resize:
            x = F.interpolate(x, (299,299), mode='bilinear', align_corners=True).type(dtype)
        x = inception_model(x)
        return F.softmax(x, 1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def cifar_runner():
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)

    cifar = dset.CIFAR10(root='data/', download=True,
                             transform=transforms.Compose([
                                 transforms.Scale(32),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ]))

    print ("Calculating Inception Score...")
    print (inception_score(IgnoreLabelDataset(cifar), cuda=True, batch_size=32, resize=True, splits=10))

def runner(path, from_pkl=True):
    if from_pkl:
        with open(os.path.join(path, 'imgs.pkl'), 'rb') as f:
            imgs = pickle.load(f)
        imgs = (imgs/255.0 - 0.5)*2 ## normalize to [-1, 1]
    else:
        imgs = list()
        for file in glob.glob(path + '/*.png'):
            img = Image.open(file)
            img = np.array(img)
            img = np.transpose(img, [2, 0, 1]) ## get shape to (3,H,W)
            img = (img/255.0 - 0.5)*2 ## normalize to [-1, 1]
            imgs.append(img)
    print('%s imgs' % len(imgs), inception_score(imgs, cuda=True, batch_size=8, resize=True, splits=10))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--from_pkl', action='store_true')
    parser.add_argument('--path', type=str, required=True)

    args = parser.parse_args()
    runner(args.path, args.from_pkl)