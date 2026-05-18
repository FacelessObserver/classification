import random
import numpy as np
import torch as t

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
t.manual_seed(SEED)
t.cuda.manual_seed(SEED)
t.backends.cudnn.deterministic = True

TEST_SIZE = 0.3
BATCH_SIZE = 16

HIDDEN_NEURONS = 20
EPOCHS = 5000
LEARNING_RATE = 0.01

CRITERION = t.nn.CrossEntropyLoss()
OPTIMIZER = t.optim.SGD
OPTIMIZER_KWARGS = {}

MODEL_PATH = "models/wine_classifier.pth"
