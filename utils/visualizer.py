import torch as t
import numpy as np

class Visualizer:
    def __init__(self, model: t.nn.Module):
        self.model = model
        self.device = "cuda" if t.cuda.is_available() else "cpu"
    
    def prepare_decision_boundary(self, x_train: t.Tensor, y_train: t.Tensor) -> dict:
        x_train_np = x_train.numpy()
        y_train_np = y_train.numpy()
        
        plot_step = 0.02
        x_min, x_max = x_train_np[:, 0].min() - 1, x_train_np[:, 0].max() + 1
        y_min, y_max = x_train_np[:, 1].min() - 1, x_train_np[:, 1].max() + 1
        
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, plot_step),
            np.arange(y_min, y_max, plot_step)
        )

        self.model.eval()
        with t.no_grad():
            mesh_points = t.FloatTensor(
                np.c_[xx.ravel(), yy.ravel()]
            ).to(self.device)
            
            logits = self.model(mesh_points)
            probs = t.nn.functional.softmax(logits, dim = 1)
            preds_class = probs.cpu().numpy().argmax(axis = 1)
            preds_class = preds_class.reshape(xx.shape)
        
        return {
            "xx": xx,
            "yy": yy,
            "preds_class": preds_class,
            "x_train": x_train_np,
            "y_train": y_train_np
        }
