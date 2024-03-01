from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import from_numpy
import os

# Custom Dataset class
class SimulationDataset(Dataset):
    def __init__(self, directory='transfer_functions', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        # Save the directories in a list for easy access
        self.data_dir = []
        for file_name in os.listdir(directory):
            if file_name.endswith('.npz'):
                file_path = os.path.join(directory, file_name)
                self.data_dir.append(file_path)

    def __len__(self):
        return len(self.data_dir)

    def __getitem__(self, idx):
        # Get the path to the file
        path = self.data_dir[idx]
        # Load the data
        data = np.load(path, allow_pickle=True)

        # Select the indeces with .speed for the generators
        var_names = data['var_names']
        indeces = [i for i, s in enumerate(var_names) if '.speed' in s]

        # Extract Transfer Function
        TF = data['TF']
        TF = TF[:, indeces]
        # Take the absolute value of the transfer function
        TF = np.abs(TF)
        # Transform to tensor
        TF = from_numpy(TF)

        # Extract frequencies
        F = data['F']
        # Reshape F to add a new dimension
        F = F.reshape(F.shape[0], 1)
        # Transform to tensor
        F = from_numpy(F)
        # Expand F to match the second dimension of TF
        F_expanded = F.expand(-1, TF.shape[1])

        # Concatenate TF and F_expanded along a new dimension
        input_data = torch.stack([TF, F_expanded], dim=2).transpose(2, 0).transpose(1, 2).to(torch.float32)  # input_data.shape = (2, 501, 10)
    

        # Extract observed momentum
        Mtot = from_numpy(data['Mtot']).to(torch.float32)

        # Apply the transformations if they exist
        if self.transform:
            input_data = self.transform(input_data)
        if self.target_transform:
            Mtot = self.target_transform(Mtot)

        # Note that:
        # input_data.shape = (number of frequencies, number of generators, 2)
        # input_data[:, :, 0] = Transfer Function
        # input_data[:, :, 1] = Frequency, repeated for each generator
        # Mtot is a scalar, the total momentum
        return input_data, Mtot
    
def get_dataloaders(directory: str, split: List[float], batch_size: float) -> Tuple[SimulationDataset, List[Dataset], List[DataLoader]]:
    dataset = SimulationDataset(directory=directory)
    generator = torch.Generator().manual_seed(42)
    splitted_sets = random_split(split, generator=generator)
    loaders = []
    for set in splitted_sets:
        loaders.append(DataLoader(set, batch_size=batch_size, shuffle=True))
    
    return (dataset, splitted_sets, loaders)
