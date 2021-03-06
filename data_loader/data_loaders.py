from base import BaseDataLoader
from data_loader.dataset import Dataset,BertDataset

class DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=64, shuffle=True, validation_split=0.0, num_workers=1, mode= "train", debug = False):

        if mode == "train":
            self.dataset = Dataset(str(data_dir/"train.pkl"), debug)
        elif mode == "test":
            batch_size = 512
            shuffle = False
            validation_split = 0.0
            self.dataset = Dataset(str(data_dir/"test.pkl"), debug)
        elif mode == "eval":
            batch_size = 512
            shuffle = False
            validation_split = 0.0
            self.dataset = Dataset(str(data_dir/"eval.pkl"), debug)

        print(f"{mode} : {self.dataset.__len__()}")
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)