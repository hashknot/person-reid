BATCH_SIZE = 100

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 50         # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.

EVAL_INTERVAL_SECS = 60
"""How often to run the eval."""

NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 800
EVAL_NUM_EXAMPLES = 800
"""Number of examples to run for eval."""

EVAL_RUN_ONCE = False
"""Whether to run eval only once."""

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 5000
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 60
IMAGE_CHANNELS = 3
NUM_CLASSES = 2
LABEL_BYTES = 1
DATA_BATCH_COUNT = 10

LOG_DEVICE_PLACEMENT = False
"""Whether to log device placement."""

LOG_FREQUENCY = 10
"""How often to log results to the console."""

TRAIN_ACCURACY_FREQUENCY = 20

USE_FP16 = False

NUM_EPOCHS = 70
MAX_STEPS = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/BATCH_SIZE * NUM_EPOCHS
"""Number of batches to run."""
