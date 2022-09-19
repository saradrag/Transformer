# training parameters
NUM_EPOCHS = 10

# model parameters
D_MODEL = 64
N = 5  # number of stacked multi head attentions
H = 4  # number of parallel multi head attentions
D_K = D_Q = D_MODEL // H
D_V = D_MODEL // H
N_WORDS_ENCODER = 1000
N_WORDS_DECODER = 1000

# dataloader parameters
BATCH_SIZE = 64

# data parameters
N_WORDS = 1000
N_TRAINING_SAMPLES = 5000
N_VALIDATION_SAMPLES = 1000
N_TEST_SAMPLES = 1000
SENTENCE_LENGTH = 10
