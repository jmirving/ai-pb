import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'resources')
DATA_FILE = 'fp-data2.csv'

# Model hyperparameters
INPUT_SIZE = 6
HIDDEN_SIZE = 128
OUTPUT_SIZE = 170
BATCH_SIZE = 6
EPOCHS = 70
LEARNING_RATE = 0.002 