import torch


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, device, eval=False):
        self.encodings = encodings
        self.device = device
        self.eval = eval

    def __getitem__(self, idx):
        if self.eval:
            return {key: val[idx] for key, val in self.encodings.items()}
        else:
            return {key: torch.tensor(val[idx]).to(self.device) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)