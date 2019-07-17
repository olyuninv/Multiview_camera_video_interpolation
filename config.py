#
# KTH Royal Institute of Technology
#

# The size of the input images to be fed to the network during training.
CROP_SIZE: int = 128

# The size of the patches to be extracted from the datasets
PATCH_SIZE = (150, 150)

# Whether or not we should store the patches produced by the data manager
CACHE_PATCHES: bool = False

# Number of epochs used for training
EPOCHS: int = 4 #16

# Kernel size of the custom Separable Convolution layer
OUTPUT_1D_KERNEL_SIZE: int = 71 #51

# The batch size used for mini batch gradient descent
BATCH_SIZE: int = 16 #32 for proper run 1

# Upper limit on the number of samples used for training
MAX_TRAINING_SAMPLES: int = 500_000

# Upper limit on the number of samples used for validation
MAX_VALIDATION_SAMPLES: int = 100

# Number of workers of the torch.utils.data.DataLoader AND of the data manager
# Set this to 0 to work on the main thread
NUM_WORKERS: int = 0

# Random seed fed to torch
SEED: int = None

# Path to the dataset directory
DATASET_DIR = '/media/lera/ADATA HV320/mv_output'

# Force torch to run on CPU even if CUDA is available
ALWAYS_CPU: bool = False

# Path to the outout directory where the model checkpoins should be stored
OUTPUT_DIR: str = './out'

# Whether or not the model parameters should be written to disk at each epoch
SAVE_CHECKPOINS: bool = True

# Force model to use the slow Separable Convolution implementation even if CUDA is available
ALWAYS_SLOW_SEP_CONV: bool = False

# Whether or not we should run the validation set on the model at each epoch
VALIDATION_ENABLED: bool = True

# Whether or not we should run the visual test set on the model at each epoch
VISUAL_TEST_ENABLED: bool = False

# Whether or not the data should be augmented with random transformations
AUGMENT_DATA: bool = True

# Probability of performing the random temporal order swap of the two input frames
RANDOM_TEMPORAL_ORDER_SWAP_PROB: float = 0.5

# Start from pre-trained model (path)
#START_FROM_EXISTING_MODEL = None
START_FROM_EXISTING_MODEL = '/home/lera/Documents/Mart_Kartasev_sepconv/src/out_2_L1_kernel71/model_epoch_26.pth'

# One of {"l1", "vgg", "ssim"}
LOSS: str = "l1"

VGG_FACTOR: float = 1.0

# One of {"None", "Add", "Replace"}
MULTI_VIEW: str = "Add"
