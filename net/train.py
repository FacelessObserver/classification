import torch as t
from .net import WineNet
from .handler import BaseModelHandler

class Trainer(BaseModelHandler):
    def __init__(self, model: WineNet):
        super().__init__(model)
        self.model.train()
    
    def train(self, criterion: t.nn.Module, dataloader: t.utils.data.DataLoader,
              epochs: int, lr: float, optimizer_class: t.optim.Optimizer,
              **optimizer_kwargs) -> dict:
        
        optimizer = optimizer_class(
            params = self.model.parameters(),
            lr = lr,
            **optimizer_kwargs
        )
        
        history = {
            "train_loss": [],
            "train_acc": []
        }
        
        for _ in range(epochs):
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += self._compute_accuracy(y_pred, batch_y)
            
            avg_loss = epoch_loss / len(dataloader)
            avg_acc = epoch_acc / len(dataloader)
            
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
        
        return history
    
    def save(self, path: str) -> bool:
        try:
            t.save(self.model.state_dict(), path)
            return True
        except Exception:
            return False
