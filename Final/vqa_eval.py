import datetime
import time
import pickle
import numpy as np
import pandas as pd

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
flags.DEFINE_string("checkpoint_dir", "/tmp/set_prediction/",
                    "Path to model checkpoint.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 10, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_bool("full_eval", True,
                  "If True, use full evaluation set, otherwise a single batch.")

def load_model():
    """Load the latest checkpoint."""
    # Build the model.
    model = model_utils.build_model(
        resolution=(128, 128), batch_size=128,
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

def load_vqa_model():
    model = Answering()
    
    # Load the weights.
    ckpt = tf.train.Checkpoint(network=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, directory="./slot_attention/tmp/set_prediction5/", max_to_keep=90)
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
    else:
        raise ValueError("Failed to load checkpoint.")
        
    return model

def run_eval(data_iterator, tokenizer, encoder, model, slot_model):
    """Run evaluation."""

    if FLAGS.full_eval:  # Evaluate on the full validation set.
        num_eval_batches = 15000 // FLAGS.batch_size
    else:
        # By default, we only test on a single batch for faster evaluation.
        num_eval_batches = 1
    
    dataframe = pd.DataFrame(columns=['question', 'answer', 'target'])
    outs = None
    num_ques = 10
    
    losses = tf.keras.metrics.CategoricalCrossentropy()
    all_questions = []
    
    for _ in tf.range(num_eval_batches):
        batch = next(data_iterator)
        
        slot_data = slot_model(batch['image'])
        
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
        
        for num in range(num_ques):
            if outs is None:
                outs = model(question[:,num,:], slot_data, training=False)
                target = answer[:,num,:]
            else:
                new_outs = model(question[:,num,:], slot_data, training=False)
                outs = tf.concat([outs, new_outs], axis=0)
                target = tf.concat([target, answer[:,num,:]], axis=0)
                
            losses.update_state(outs, target)
         
        all_questions.extend(decode_questions)
        
    accuracy = tf.keras.metrics.Accuracy()
    dataframe = pd.DataFrame()
    
    print(tf.shape(target))
    print(tf.shape(outs))
    
    accuracy.update_state(tf.math.argmax(target, axis=1).numpy(), tf.math.argmax(outs, axis=1).numpy())
            
    logging.info("Finished getting model predictions.")
    
    
    dataframe['question'] = all_questions
    dataframe['answer'] = encoder.inverse_transform(tf.math.argmax(outs, axis=1).numpy().tolist())
    dataframe['target'] = encoder.inverse_transform(tf.math.argmax(target, axis=1).numpy().tolist())
    
    dataframe.to_csv("./slot_attention/dataframe/predict_target.csv", index=False)

    return losses, accuracy

def main(argv):
    del argv
    
    slot_model = load_model()
    
    model = load_vqa_model()
    
    dataset = data_utils.build_clevr_iterator(
        batch_size=FLAGS.batch_size, split="validation", resolution=(128, 128))
    
    with open('./tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
            
    with open('./answer_encoder.pickle', 'rb') as handle:
        encoder = pickle.load(handle)

    loss, acc = run_eval(dataset, tokenizer, encoder, model, slot_model)

    logging.info(
        "validation loss: %.2f, validation accuracy: %.2f", loss.result().numpy(), acc.result().numpy())


if __name__ == "__main__":
    app.run(main)