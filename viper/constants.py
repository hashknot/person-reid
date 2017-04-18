BATCH_SIZE = 200

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 200.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.001     # Initial learning rate.

MAX_STEPS = 1000000
"""Number of batches to run."""

EVAL_INTERVAL_SECS = 60
"""How often to run the eval."""

EVAL_NUM_EXAMPLES = 500
"""Number of examples to run for eval."""

EVAL_RUN_ONCE = False
"""Whether to run eval only once."""

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 2000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 500
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 48
IMAGE_CHANNELS = 3
NUM_CLASSES = 2
LABEL_BYTES = 1

LOG_DEVICE_PLACEMENT = False
"""Whether to log device placement."""

LOG_FREQUENCY = 10
"""How often to log results to the console."""

USE_FP16 = False
