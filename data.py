import numpy as np
from parameters import N_WORDS, N_TRAINING_SAMPLES, N_VALIDATION_SAMPLES, N_TEST_SAMPLES, SENTENCE_LENGTH

# run this once
training_arrays = np.random.randint(N_WORDS, size=(N_TRAINING_SAMPLES, SENTENCE_LENGTH))
validation_arrays = np.random.randint(N_WORDS, size=(N_VALIDATION_SAMPLES, SENTENCE_LENGTH))
test_arrays = np.random.randint(N_WORDS, size=(N_TEST_SAMPLES, SENTENCE_LENGTH))
training_labels = np.sort(training_arrays)
validation_labels = np.sort(validation_arrays)
test_labels = np.sort(test_arrays)
np.savetxt('data/training_arrays.txt', fmt='%1i', X=training_arrays)
np.savetxt('data/validation_arrays.txt', fmt='%1i', X=validation_arrays)
np.savetxt('data/test_arrays.txt', fmt='%1i', X=test_arrays)
np.savetxt('data/training_labels.txt', fmt='%1i', X=training_labels)
np.savetxt('data/validation_labels.txt', fmt='%1i', X=validation_labels)
np.savetxt('data/test_labels.txt', fmt='%1i', X=test_labels)
