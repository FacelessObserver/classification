import torch as t

class BaseModelHandler:
    def __init__(self, model: t.nn.Module):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
    
    def _compute_accuracy(self, y_pred: t.Tensor, y_true: t.Tensor) -> float:
        _, predicted = t.max(y_pred, 1)
        correct = (predicted == y_true).sum().item()
        return correct / len(y_true)
