import torch as t
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

class Data:
    def __init__(self):
        self.x_train: t.Tensor | None = None
        self.x_test: t.Tensor | None = None
        
        self.y_train: t.Tensor | None = None
        self.y_test: t.Tensor | None = None

        self.feature_names = None
        self.class_names = None

    def load(self, test_size: float):
        dataset = load_wine()

        x = dataset.data[:, :2]
        y = dataset.target

        self.feature_names = dataset.feature_names[:2]
        self.class_names = dataset.target_names.tolist()

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size = test_size,
            shuffle = True
        )

        self.x_train = t.FloatTensor(x_train)
        self.x_test = t.FloatTensor(x_test)
        self.y_train = t.LongTensor(y_train)
        self.y_test = t.LongTensor(y_test)
    
    def get_loaders(self, batch_size: int) -> tuple[DataLoader]:
        train_dataloader = DataLoader(
            dataset = TensorDataset(
                self.x_train,
                self.y_train
            ),
            batch_size = batch_size,
            shuffle = True
        )

        test_dataloader = DataLoader(
            dataset = TensorDataset(
                self.x_test,
                self.y_test
            ),
            batch_size = batch_size,
            shuffle = False
        )

        return train_dataloader, test_dataloader
