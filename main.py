import os
from utils.data import Data
from net.net import WineNet
from net.train import Trainer
from net.test import Tester
from utils.drawer import Drawer
from utils.visualizer import Visualizer
from config import *

def main():
    os.makedirs("models", exist_ok = True)
    os.makedirs("plots", exist_ok = True)
    
    print("Загрузка данных")
    data = Data()
    data.load(test_size = TEST_SIZE)
    train_loader, test_loader = data.get_loaders(batch_size = BATCH_SIZE)

    model = WineNet(hidden_neurons = HIDDEN_NEURONS)
    print(f"Модель создана с {HIDDEN_NEURONS} нейронами в скрытом слое")

    print("Производится обучение")
    trainer = Trainer(model)
    history = trainer.train(
        criterion = CRITERION,
        dataloader = train_loader,
        epochs = EPOCHS,
        lr = LEARNING_RATE,
        optimizer_class = OPTIMIZER,
        **OPTIMIZER_KWARGS
    )

    trainer.save(MODEL_PATH)

    Drawer.plot_training_history(
        history["train_loss"], 
        history["train_acc"],
        save_path = "plots/training_history.png"
    )

    print("Производится тестирование")  
    tester = Tester(model)
    metrics = tester.test(
        criterion = CRITERION,
        dataloader = test_loader
    )

    print(f"\nИтоговые метрики")
    print(f"Loss: {metrics['loss']:.6f}")
    print(f"Acc: {metrics['accuracy']:.6f}")
    
    visualizer = Visualizer(model)
    boundary_data = visualizer.prepare_decision_boundary(
        data.x_train,
        data.y_train
    )

    Drawer.plot_decision_boundary(
        **boundary_data,
        class_names = data.class_names,
        feature_names = data.feature_names,
        save_path = "plots/decision_boundary.png"
    )

if __name__ == "__main__":
    main()
