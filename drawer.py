import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional

class Drawer:
    
    matplotlib.rcParams["figure.figsize"] = (10, 8)

    @staticmethod
    def plot_training_history(
        losses: List[float],
        accuracies: List[float],
        save_path: Optional[str] = None
        ):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (13, 5))
        
        ax1.plot(losses, "b-", linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha = 0.3)
        
        ax2.plot(accuracies, "g-", linewidth = 2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.grid(True, alpha = 0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    @staticmethod
    def plot_decision_boundary(
        xx: np.ndarray, yy: np.ndarray, 
        preds_class: np.ndarray,
        x_train: np.ndarray, y_train: np.ndarray,
        class_names: List[str] = None,
        feature_names: List[str] = None,
        save_path: Optional[str] = None
        ):
        
        plot_colors = ["green", "orange", "black"]
        n_classes = len(np.unique(y_train))
        
        plt.figure(figsize = (10, 8))
        plt.contourf(xx, yy, preds_class, cmap = "Accent", alpha = 0.6)
        
        for i, color in zip(range(n_classes), plot_colors[:n_classes]):
            indexes = np.where(y_train == i)
            plt.scatter(
                x_train[indexes, 0],
                x_train[indexes, 1],
                c = color,
                label = class_names[i] if class_names else f"Class {i}",
                edgecolors = "black",
                s = 50
            )
        
        if feature_names:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
        
        plt.legend()
        plt.title("Decision Boundary")
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
