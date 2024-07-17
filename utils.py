import numpy as np
import gdown
import os
import torch


def get_dataset():

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



def reshape_data(data_path:str, number_degradations:int):

    '''
    12 Degradations 
    1 Ground Truth
    '''

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