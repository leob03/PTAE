import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset

class CPP_Dataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, label: bool = True,
                 padding: int = None, transform = None) -> None:
        self.root_dir = root_dir
        self.annotations = pd.read_csv(f'{root_dir}/{csv_file}')
        self.label, self.padding, self.transform = label, padding, transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx, 0]
        data_path = os.path.join(self.root_dir, item)
        label = item.split('/', 1)[0]

        pts = np.load(data_path)
        num_pts = pts.shape[0]

        if self.transform is not None:
            pts = self.transform(pts)

        if self.padding is not None and num_pts < self.padding:
            num_pad = self.padding - num_pts
            pts = np.pad(pts, [(0,num_pad),(0,0)], 'constant')

        return (pts, num_pts, label) if self.label else (pts, num_pts)

# Original implementation from https://github.com/XuyangBai/FoldingNet/blob/master/loss.py

class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)

        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        _, num_points_x, _ = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))

        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
            
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)

        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)

        return P

def gen_csv(root_dir: str) -> None:
    """ 
    Generate summary of the dataset in .csv format.
    
    Parameters
    ----------
    root_dir : str
        root directory of the dataset
        
    Returns
    -------
    None
    """
    with open(f'{root_dir}/summary.csv','w') as summary:
        summary.write('Data')
        summary.write('\n')
        for folder in os.scandir(root_dir):
            if folder.is_dir():
                sort_paths = sorted(os.listdir(folder.path))
                for file in sort_paths:
                    item = f'{folder.name}/{file}'
                    summary.write(item)
                    summary.write('\n')
            else:
                print(f'Not a directory: {folder.name}')
        summary.close()

def visualizer(epoch: int, prediction: list, save_dir: str = None) -> None:
    """
    Visualize predictions and save them
    """
