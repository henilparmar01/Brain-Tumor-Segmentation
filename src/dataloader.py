from torch.utils.data import DataLoader
from dataset import BrainTumorDataset

def create_dataloader():

    dataset = BrainTumorDataset()

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=2
    )

    return dataloader

def main():
    pass
   

if __name__ == "__main__":
    main()