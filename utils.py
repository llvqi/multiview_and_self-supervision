import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import glob
import PIL

# Shuffle the private information
def sample(n1=183*2, n2=24):
    idx = []
    for i in range(n1):
        idx.append(np.random.permutation(n2)+i*n2)

    return np.hstack(idx)


# Multiview Dateset
class ViewDataset(Dataset):
    def __init__(self, v1, v2):
        self.v1 = torch.tensor(v1).float()
        self.v2 = torch.tensor(v2).float()
        self.data_len = v1.shape[0]

    def __getitem__(self, index):
        return self.v1[index], self.v2[index], index

    def __len__(self):
        return self.data_len


# Get a dataloader
def get_dataloader(view1, view2, batchsize, shuffle):
    dataset = ViewDataset(view1, view2)

    # Dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                            batch_size=batchsize,
                                            shuffle=shuffle)

    return data_loader


# Get the Car3D dataset
def get_cars3d(filedir='./data/cars/'):
    tmp = []
    for f in glob.glob(filedir+'*mesh.mat'):
        a = sio.loadmat(f)
        tt = np.zeros((a['im'].shape[3], a['im'].shape[4], 64, 64, 3))
        for i in range(a['im'].shape[3]):
            for j in range(a['im'].shape[4]):
                pic = PIL.Image.fromarray(a['im'][:,:,:,i,j])
                pic.thumbnail((64, 64, 3), PIL.Image.ANTIALIAS)
                tt[i, j, :, :, :] = np.array(pic)/255.

        b = torch.tensor(tt)
        c = b.permute(0,1,4,2,3)
        tmp.append(c)

    data = torch.stack(tmp, dim=0)


    a, b, c = data.shape[:3]

    imgs = data.numpy()
    imgs = np.transpose(imgs, (2,0,1,3,4,5))

    # 4 elevations
    elv1 = imgs[0]
    elv2 = imgs[1]
    elv3 = imgs[2]
    elv4 = imgs[3]

    # Show samples
    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(np.transpose(elv1[0,1,:,:,:],(1,2,0)))
    axarr[0].axis('off')
    axarr[1].imshow(np.transpose(elv2[0,7,:,:,:],(1,2,0)))
    axarr[1].axis('off')
    axarr[2].imshow(np.transpose(elv3[0,14,:,:,:],(1,2,0)))
    axarr[2].axis('off')
    axarr[3].imshow(np.transpose(elv4[0,21,:,:,:],(1,2,0)))
    axarr[3].axis('off')

    fig.suptitle('Samples of both views, left two for view1, right two for view2',fontsize=10)
    plt.show()

    # Get the two views in order
    view1=[]
    view2=[]

    for i in range(elv1.shape[0]):
        # Lower elevations
        view1.append(elv1[i,:,:,:,:])
        view1.append(elv2[i,:,:,:,:])
        # Higher elevations
        view2.append(elv3[i,:,:,:,:])
        view2.append(elv4[i,:,:,:,:])

    view1 = np.concatenate(view1, axis=0)
    view2 = np.concatenate(view2, axis=0)

    return view1, view2
