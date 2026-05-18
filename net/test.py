import torch as t
from .handler import BaseModelHandler
from .net import WineNet

class Tester(BaseModelHandler):
    def __init__(self, model: WineNet):
        super().__init__(model)
        self.model.eval()
    
    def test(self, criterion: t.nn.Module, 
             dataloader: t.utils.data.DataLoader) -> dict:

        test_loss = 0
        predictions = []
        targets = []
        
        with t.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                y_pred = self.model(batch_x)
                loss = criterion(y_pred, batch_y)
                
                test_loss += loss.item()
                predictions.append(y_pred.cpu())
                targets.append(batch_y.cpu())

        predictions = t.cat(predictions)
        targets = t.cat(targets)

        avg_loss = test_loss / len(dataloader)
        accuracy = self._compute_accuracy(predictions, targets)
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
        }
