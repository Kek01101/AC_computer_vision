import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import os
import sys
import time
import string
from copy import deepcopy

# NOTE: Model directory path needs to include the model name listed in the checkpoint.txt file
"""
model_dir = directory where model is kept - see note above ex: direction_final/direction_final_model
data_dir = directory where the dataset is kept - needs to be loaded in to allow for proper predicting
epochs = number of epochs model will train for
"""
if len(sys.argv) not in [2, 3, 4]:
    sys.exit("Usage: python final_predicting [model_dir] [data_dir] [output_name]")

locations = str(sys.argv[2])
output_name = str(sys.argv[3])

"""
Normally, this section would be where all the models are defined. However, that has proven to break the code. 
Therefore, this script is much messier than the training script.
"""

# Import dataset
def load_data(location):

    # List for holding the sets from the entirety of the dataset
    sets = []

    # Loading all the sets from the dataset
    for file in os.listdir(location):
        with open(os.path.join(location, file), "r+", encoding="UTF-8") as f:
            sets.append(f.read())

    return sets


sets = load_data(locations)
# Convert sentences into a single string but with each line being a different review
work_set = ""
# Running through every
for a in sets:
    work_set += str(a + "\n")
vocab = sorted(set(work_set))
# Ensuring that no non-digit chars have managed to get into the dataset
accepted_chars = [" ", "\n"]
for digit in string.digits:
    accepted_chars.append(digit)
for char in deepcopy(vocab):
    if char not in accepted_chars:
        vocab.remove(char)

chars = tf.strings.unicode_split(work_set, input_encoding="UTF-8")
# Maps string forms of ints to actual ints for use by the model
ids_from_chars = preprocessing.StringLookup(
    vocabulary=list(vocab)
)

# Converts between str and int form of the ints in the dataset
def text_from_ids(ids):
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)


chars_from_ids = preprocessing.StringLookup(vocabulary=ids_from_chars.get_vocabulary(), invert=True)

# Slicing up the int ids into datasets
all_ids = ids_from_chars(chars)
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)

# Sequence length is how many ints are in a training/testing sequence - 100 seems to work well
seq_length = 100
examples_per_epoch = len(work_set)//(seq_length+1)

# Converting list of ints into desired sequence lengths
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)


# Splits dataset into input and target sets for training
def split_input_target(sequence):
    input_set = sequence[:-1]
    target_set = sequence[1:]
    return input_set, target_set


# Finally creates the actual dataset
dataset = sequences.map(split_input_target)

# Training batches
"""
Dataset needs to be split into batches and shuffled in order for training to function correctly.
"""

# DO NOT USE VARIABLES FOR RESPRESNTING SHUFFLE BUFFER OR BATCH SIZE - BREAKS EVERYTHING
# Dataset batching and shuffling and prefetching
dataset = dataset.shuffle(10000)
dataset = dataset.batch(64)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


# Define model here
# Output layer size - equivalent to vocab size - so 3
vocab_size = len(vocab)

# Embedding dimension - Input layer - maps vectors
embedding_dim = 256

# Number of RNN units - middle layer
rnn_units = 1024


# Model here does not need to include the custom training model for predicting, weights are all that matter
class MyModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True)
    self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs
    x = self.embedding(x, training=training)
    if states is None:
      states = self.gru.get_initial_state(x)
    x, states = self.gru(x, initial_state=states, training=training)
    x = self.dense(x, training=training)

    if return_state:
      return x, states
    else:
      return x


# Model object created - saved to var model
model = MyModel(
    vocab_size=len(ids_from_chars.get_vocabulary()),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units)

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss)
location = str(sys.argv[1])
model.load_weights(location)

# This function performs the actual predicting function - feeds each int into the function to determine the next int
# Each following int is again fed into the function until the desired length is met
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.ids_from_chars(['', '[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices=skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs=input_ids, states=states,
                                              return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states


# Building the predicting model after definition
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

# Actually running the model
start = time.time()
states = None
# String of no inputs is used as the starting value, mimicks what the dataset looked like - 5 sets made at a time
next_char = tf.constant(["00000000000000000000000000000000000000000000000000000000000000"]*5)
result = [next_char]

for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result)
end = time.time()
for a in range(len(result)):
    print(result[a].numpy().decode('utf-8'), '\n\n' + '_'*80)
print('\nRun time:', end - start)
# You may want to change this output file to a full path depending on your system
with open(f"{output_name}.txt", "w", encoding="utf-8") as file:
    for a in range(len(result)):
        file.write(f"Set {a+1}:\n")
        file.write(str(result[a].numpy().decode("utf-8")))
        file.write("\n")
        file.write("\n")