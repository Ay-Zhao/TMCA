import torch
from torch.utils.data import Dataset

class MoleculeDataset(Dataset):
    def __init__(self, data):
        """
        data_list: 一个包含多个样本的列表，每个样本是一个字典
        unimol
        {
            'src_tokens': numpy array or tensor,
            'src_distance': numpy array or tensor,
            'src_coord': numpy array or tensor,
            'src_edge_type': numpy array or tensor,
        }
        """

        self.x = data['unimol_input']
        self.y = data['target']

        # flatten to scalar labels: e.g. [[1],[0],...] -> [1,0,...]
        # keep them as python int/float; collate will torch.tensor([...]) later
        self.y = [int(v[0]) if isinstance(v, (list, tuple)) or getattr(v, "ndim", 0) == 1 else int(v)
                  for v in self.y]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        # turn arrays -> tensors with correct dtypes
        sample_dict = {
            "src_tokens":    torch.as_tensor(x['src_tokens'],    dtype=torch.long),
            "src_distance":  torch.as_tensor(x['src_distance'],  dtype=torch.float32),
            "src_coord":     torch.as_tensor(x['src_coord'],     dtype=torch.float32),
            "src_edge_type": torch.as_tensor(x['src_edge_type'], dtype=torch.long),
        }
        label = self.y[idx]  # scalar int (0/1)
        return sample_dict, label


