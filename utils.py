import numpy as np
import gdown
import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
import torchvision.utils as vutils



def get_weights():
    if os.path.exists('weights/drunet_color.pth'):
        print('Weights already downloaded!')
        return
    os.makedirs('weights', exist_ok=True)
    model_url= 'https://huggingface.co/deepinv/drunet/resolve/main/drunet_color.pth?download=true'
    torch.hub.download_url_to_file(model_url, "weights/drunet_color.pth", hash_prefix=None, progress=True)


def get_dataset_seismic():

    if os.path.exists('data/dataset.npy'):
        print('Dataset already downloaded!')
        return
    
    shared_url = 'https://drive.google.com/file/d/1No-D6AOIHfXXpuh4AZVHfIjiP4Imo69T/view?usp=drive_link'
    file_id = shared_url.split('/d/')[1].split('/view')[0]
    direct_url = f'https://drive.google.com/uc?id={file_id}'
    output_path = 'data/'
    os.makedirs(output_path, exist_ok=True)
    gdown.download(direct_url, f'{output_path}/dataset.npy', quiet=False)

    print('Dataset downloaded successfully!')


def get_validation_set(dst_train, split=0.1):

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split = int(np.floor(split * len(dst_train)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)

    return train_sample, val_sample



def get_dataset_torch(dataset:str, data_path:str, batch_size:int):

    if dataset == 'MNIST':
        channel = 1
        num_classes = 10
        im_size = (28, 28)
        transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)  
        dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
        train_sample, val_sample = get_validation_set(dst_train, split=0.1)
        class_names = [str(c) for c in range(num_classes)]

    elif dataset == 'FMNIST':
        channel = 1
        num_classes = 10
        im_size = (28, 28)
        transform = transforms.Compose([transforms.ToTensor()])
        dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)  
        dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
        train_sample, val_sample = get_validation_set(dst_train, split=0.1)
        class_names = dst_train.classes


    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, num_workers=0, sampler=train_sample)
    valoader = torch.utils.data.DataLoader(dst_train, batch_size=batch_size, num_workers=0, sampler=val_sample)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=batch_size, shuffle=False, num_workers=0)

    return channel, im_size, num_classes, class_names, dst_train, dst_test, testloader, trainloader, valoader  


def reshape_data(data_path:str, number_degradations:int):

    '''
    12 Degradations 
    1 Ground Truth
    Images of 256x256
    '''

    if os.path.exists('data/dataset_reshaped.npy'):
        print('Data already reshaped!')
        return

    data = np.load(data_path, allow_pickle=True)
    data = data[:, :number_degradations + 1]
    data = [[data[i,j] for j in range(number_degradations + 1)] for i in range(len(data))]
    data = np.array(data) 

    data = torch.tensor(data, dtype=torch.float32)
    reshaped_data = data[:, 1:].reshape(-1, 256, 256)
    ground_truths = data[:, 0].unsqueeze(1).repeat(1, number_degradations, 1, 1).reshape(-1, 256, 256)

    new_data = torch.stack((ground_truths, reshaped_data), dim=1).numpy()

    with open(f'data/dataset_reshaped.npy', 'wb') as f:
        np.save(f, new_data)

    print('Data reshaped successfully to shape:', new_data.shape)


class SeismicDataset(Dataset):

    def __init__(self, data_path:str, transform=None):
        self.data = np.load(data_path, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample = self.data[idx]
        ground_truth = sample[0]
        degraded = sample[1]

        if self.transform:
            ground_truth = self.transform(ground_truth)
            degraded = self.transform(degraded)

        return ground_truth, degraded
    


def get_test_val_set(dst_train, split_test = 0.1, split_val = 0.1):

    indices = list(range(len(dst_train)))
    np.random.shuffle(indices)
    split_test = int(np.floor(split_test * len(dst_train)))
    split_val = int(np.floor(split_val * len(dst_train)))
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[split_test+split_val:], indices[:split_val], indices[split_val:split_test+split_val]

    train_sample = SubsetRandomSampler(train_indices)
    val_sample = SubsetRandomSampler(val_indices)
    test_sample = SubsetRandomSampler(test_indices)

    return train_sample, val_sample, test_sample
    

def get_dataloader(batch_size:int, num_workers:int, data_path:str, im_size: tuple):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(im_size)])
    dataset = SeismicDataset(data_path, transform=transform)
    train_sample, val_sample, test_sample = get_test_val_set(dataset)
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sample, num_workers=num_workers)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sample, num_workers=num_workers)
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sample, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def save_reconstructed_images(imgs, recons, num_img, pad, path, name, PSNR, SSIM):

    grid = vutils.make_grid(torch.cat((imgs[:num_img], recons[:num_img])), nrow=num_img, padding=pad, normalize=True)
    vutils.save_image(grid, f'{path}/{name}.png')

    psnr_imgs = [np.round(PSNR(recons[i].unsqueeze(0), imgs[i].unsqueeze(0)).item(),2) for i in range(num_img)]
    ssim_imgs = [np.round(SSIM(recons[i].unsqueeze(0), imgs[i].unsqueeze(0)).item(),3) for i in range(num_img)]

    return grid, psnr_imgs, ssim_imgs


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_metrics(save_path):

    images_path = save_path + '/images'
    model_path = save_path + '/model'
    metrics_path = save_path + '/metrics'

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    return images_path, model_path, metrics_path


def save_npy_metric(file, metric_name):

    with open(f'{metric_name}.npy', 'wb') as f:
        np.save(f, file)