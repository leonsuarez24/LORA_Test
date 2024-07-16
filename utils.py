import numpy as np
import gdown
import os


def get_dataset():
    
    shared_url = 'https://drive.google.com/file/d/1No-D6AOIHfXXpuh4AZVHfIjiP4Imo69T/view?usp=drive_link'
    file_id = shared_url.split('/d/')[1].split('/view')[0]
    direct_url = f'https://drive.google.com/uc?id={file_id}'
    output_path = 'data/'
    os.makedirs(output_path, exist_ok=True)
    gdown.download(direct_url, f'{output_path}/dataset.npy', quiet=False)



def reshape_data(data_path:str, number_degradations:int):

    data = np.load(data_path, allow_pickle=True)
    data = data[:, :number_degradations]
    data_2 = [[data[i,j] for j in range(number_degradations)] for i in range(len(data))]
    reshaped_data = np.array(data_2)
    with open(f'dataset_reshaped_{number_degradations}.npy', 'wb') as f:
        np.save(f, reshaped_data)