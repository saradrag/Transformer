# training parameters
NUM_EPOCHS = 10

# model parameters
D_MODEL = 512
N = 5  # number of stacked multi head attentions
H = 4  # number of parallel multi head attentions
D_K = D_Q = D_MODEL // H
D_V = D_MODEL // H
N_DECODER_OUTPUT = 33
P_DROPOUT = 0.1

# dataloader parameters
BATCH_SIZE = 64

