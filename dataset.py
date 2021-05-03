import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, device):
        self.encodings = encodings
        self.device = device

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).to(self.device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)