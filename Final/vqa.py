import datetime
import time
import pickle
import numpy as np

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras import Input

import slot_attention.data as data_utils
import slot_attention.model as model_utils
import slot_attention.utils as utils

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "./slot_attention/tmp/vqa/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 128, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.00005, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 1000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 10000,
                     "Number of steps for the learning rate decay.")


class Answering(tf.keras.Model):
    def __init__(self):
        super().__init__()
        
        self.encoder = Sequential([
            layers.Embedding(81,32),
            layers.LSTM(64, return_sequences=True),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(128, return_sequences=True),
            layers.LSTM(64),
#             layers.LSTM(64),
#             layers.Dense(128),
#             layers.Dense(256),
#             layers.Dense(128),
#             layers.Dense(64),
            layers.Dense(19)
        ])
        
        self.dense = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(1024, activation='relu')
        self.dense3 = layers.Dense(1024, activation='relu')
        self.dense4 = layers.Dense(512, activation='relu')
#         self.dense5 = layers.Dense(256, activation='relu')
#         self.dense6 = layers.Dense(64, activation='relu')
        self.fc = layers.Dense(29, activation='softmax')

    def call(self, x, slot_data):
        x = self.encoder(x) ## (64, 19)
        x = tf.expand_dims(x, axis=1) ## (64, 1, 19)
#         x = tf.math.multiply(slot_data, x) ## (64, 10, 19)
        x = tf.concat([slot_data, x], axis=1) ## (64, 11, 19)
        x = layers.Flatten()(x) ## (64, 190)

        h = self.dense(x) ## (64, 128)
        h = self.dense2(h)
        h = self.dense3(h)
        h = self.dense4(h)
#         h = self.dense5(h)
#         h = self.dense6(h)
        out = self.fc(h)
        
        return out
    
def vqa_train_step(batch, tokenizer, encoder, model, optimizer, slot_data, losses, accuracy):
    
    questions = batch['question']
    decode_questions = [ques.decode('ascii') for ques in questions.numpy().flatten().tolist()]
    sequences = tokenizer.texts_to_sequences(decode_questions)

    word_index = tokenizer.word_index
    question = pad_sequences(sequences, maxlen=50).reshape(questions.shape[0], questions.shape[1], 50) ##
    question = tf.convert_to_tensor(question, dtype=tf.float32) ## (64, 10, 50)
    
    answers = batch['answer']
    decode_answers = [anw.decode('ascii') for anw in answers.numpy().flatten().tolist()]
    
    answer = encoder.transform(decode_answers).reshape(answers.shape[0], answers.shape[1]) ## (None, 28)
#     answer = tf.convert_to_tensor(answer, dtype=tf.float32) ## (64, 10, 28)
    answer = tf.one_hot(answer, 29)
#     num = np.random.randint(0, 10)
    num = np.random.randint(10)
    
    with tf.GradientTape() as tape:
        preds = model(question[:,num,:], slot_data, training=True)
        loss_value = tf.keras.losses.categorical_crossentropy(answer[:,num,:], preds)
        losses.update_state(answer[:,num,:], preds)
        accuracy.update_state(tf.math.argmax(preds, axis=1).numpy(), tf.math.argmax(answer[:,num,:], axis=1).numpy())

    # Get and apply gradients.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    return losses, accuracy

def vqa_val_step(batch, tokenizer, encoder, model, slot_data, losses, accuracy):
    
    questions = batch['question']
    decode_questions = [ques.decode('ascii') for ques in questions.numpy().flatten().tolist()]
    sequences = tokenizer.texts_to_sequences(decode_questions)

    word_index = tokenizer.word_index
    question = pad_sequences(sequences, maxlen=50).reshape(questions.shape[0], questions.shape[1], 50) ##
    question = tf.convert_to_tensor(question, dtype=tf.float32) ## (64, 10, 50)
    
    answers = batch['answer']
    decode_answers = [anw.decode('ascii') for anw in answers.numpy().flatten().tolist()]
    
    answer = encoder.transform(decode_answers).reshape(answers.shape[0], answers.shape[1]) ## (None, 28)
    answer = tf.one_hot(answer, 29)
    num_ques = 10
    
    for num in range(num_ques):
        preds = model(question[:,num,:], slot_data, training=True)
        losses.update_state(answer[:,num,:], preds)
        accuracy.update_state(tf.math.argmax(preds, axis=1).numpy(), tf.math.argmax(answer[:,num,:], axis=1).numpy())

    return losses, accuracy
    
def load_model():
    """Load the latest checkpoint."""
    # Build the model.
    model = model_utils.build_model(
        resolution=(128, 128), batch_size=FLAGS.batch_size,
        num_slots=FLAGS.num_slots, num_iterations=FLAGS.num_iterations,
        model_type="set_prediction")
    
    # Load the weights.
    ckpt = tf.train.Checkpoint(network=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory="/tmp/set_prediction/", max_to_keep=5)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
    else:
        raise ValueError("Failed to load checkpoint.")
    return model

def main(argv):
    del argv
    # Hyperparameters of the model.
    batch_size = FLAGS.batch_size
    num_slots = FLAGS.num_slots
    num_iterations = FLAGS.num_iterations
    base_learning_rate = FLAGS.learning_rate
    num_train_steps = FLAGS.num_train_steps
    warmup_steps = FLAGS.warmup_steps
    decay_rate = FLAGS.decay_rate
    decay_steps = FLAGS.decay_steps
    tf.random.set_seed(FLAGS.seed)
    resolution = (128, 128)
    
      # Build dataset iterators, optimizers and model.
    data_iterator = data_utils.build_clevr_iterator(
        batch_size, split="train", resolution=resolution, shuffle=True,
        max_n_objects=10, get_properties=True, apply_crop=False)

    data_iterator_validation = data_utils.build_clevr_iterator(
        batch_size, split="train_eval", resolution=resolution, shuffle=False,
        max_n_objects=10, get_properties=True, apply_crop=False)
    
    optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)
    slot_model = load_model()
    
    model = Answering()
    
     # Prepare checkpoint manager.
    global_step = tf.Variable(0, trainable=False, name="global_step", dtype=tf.int64)
    
    ckpt = tf.train.Checkpoint(
        network=model, optimizer=optimizer, global_step=global_step)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=100)
    ckpt.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
    else:
        logging.info("Initializing from scratch.")

    start = time.time()
    
    with open('./tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
            
    with open('./answer_encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    for _ in range(num_train_steps):
        losses = tf.keras.metrics.CategoricalCrossentropy()
        accuracy = tf.keras.metrics.Accuracy()
        batch = next(data_iterator)

        # Learning rate warm-up.
        if global_step < warmup_steps:
            learning_rate = base_learning_rate * tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
        else:
            learning_rate = base_learning_rate
        
        learning_rate = base_learning_rate
        
        learning_rate = learning_rate * (decay_rate ** (tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
        optimizer.lr = learning_rate.numpy()

        slot_data = slot_model(batch['image']) ### (64, 10, 19)

        vqa_loss, vqa_acc = vqa_train_step(batch, tokenizer, encoder, model, optimizer, slot_data, losses, accuracy)
        
#         # Update the global step. We update it before logging the loss and saving
#         # the model so that the last checkpoint is saved at the last iteration.
        global_step.assign_add(1)

#         # Log the training loss and validation average precision.
#         # We save the checkpoints every 1000 iterations.
        if not global_step % 100:
            logging.info("Step: %s, Loss: %.6f, acc: %.6f, Time: %s",
                         global_step.numpy(), vqa_loss.result().numpy(), vqa_acc.result().numpy(),
                         datetime.timedelta(seconds=time.time() - start))
        if not global_step  % 1000:
#             # For evaluating the AP score, we get a batch from the validation dataset.
            losses = tf.keras.metrics.CategoricalCrossentropy()
            accuracy = tf.keras.metrics.Accuracy()
        
            batch = next(data_iterator_validation)
            slot_data = slot_model(batch['image'])
            
            val_loss, val_acc = vqa_val_step(batch, tokenizer, encoder, model, slot_data, losses, accuracy)

            logging.info(
                "validation loss: %.2f, validation accuracy: %.2f", val_loss.result().numpy(), val_acc.result().numpy())

            # Save the checkpoint of the model.
            saved_ckpt = ckpt_manager.save()
            logging.info("Saved checkpoint: %s", saved_ckpt)

if __name__ == "__main__":
    app.run(main)
