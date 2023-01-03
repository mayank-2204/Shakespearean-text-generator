# import the streamlit library
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import numpy as np
# from tensorflow.keras.saving import pickle_utils

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout,GRU

import pickle

path_to_file = '/Users/mayank/Documents/NLP_Project/shakespeare.txt'

text = open(path_to_file,'r').read()

vocab = sorted(set(text)) # gives us all the unique characters


char_to_ind = {char:ind for ind,char in enumerate(vocab)}
ind_to_char = np.array(vocab)

encoded_text = np.array([char_to_ind[c] for c in text])
# this line will encode everything in the shakespear text file into a numpy array AKA NUMERIC VALUE FOR EVERY WORD.

# we take the first 3 lines and check their length
lines = '''
From fairest creatures we desire increase,
  That thereby beauty's rose might never die,
  But as the riper should by time decease,
'''
len(lines)

seq_len = 120 # WE TAKE SEQUENCE LENGTH OF 120 BECAUSE THAT SEEMS SUFFICIENT FOR THE MODEL TO UNDERSTAND THE TEXT DATA AND MAKE ACCURATE PREDICTIONS.

total_num_seq = len(text) // (seq_len+1)


char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

for item in char_dataset.take(500): 
  print(ind_to_char[item.numpy()]) #this returns every alphabet individually from every word. 

# we create our custom loss function here to make sure from_logits=True bcz it should be true when we are ONE-HOT ENCODED WHICH WE ARE.
def sparse_cat_loss(y_true,y_pred):
  return sparse_categorical_crossentropy(y_true,y_pred, from_logits=True)


def create_model(vocab_size,embed_dim,rnn_neurons,batch_size):

  model = Sequential()
  model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, batch_input_shape=[batch_size,None]))

  model.add(GRU(units=rnn_neurons, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'))
  #NOW WE ADD OUR FINAL DENSE LAYER:
  model.add(Dense(vocab_size))

  model.compile('adam', loss=sparse_cat_loss)

  return model

# open a file, where you stored the pickled data
model = create_model(84, 64, 1026, batch_size=1)

model.load_weights('shakespeare_gen.h5') # we don't load the whole model, we just use its weights and train the model ourselves

model.build(tf.TensorShape([1,None]))

# with open('Script' , 'rb') as f:
#    model = pickle.load(f)



# give a title to our app
st.title('Welcome to Shakesphere Script Generator')

# radio button to choose height format
character_name = st.radio('Select your Character: ',
				('Romeo', 'Juliet','Tybalt','Friar Laurence','Mercutio','Benvolio','Capulet','Count Paris'))

words = st.number_input('Number of letter', min_value=1, max_value=10000, step=1)

def generate_text(model, start_seed, gen_size, temp=1.0):
  '''
  model: Trained Model to Generate Text
  start_seed: Intial Seed text in string form
  gen_size: Number of characters to generate

  Basic idea behind this function is to take in some seed text, format it so
  that it is in the correct shape for our network, then loop the sequence as
  we keep adding our own predicted characters. Similar to our work in the RNN
  time series problems.
  '''
  # Number of characters to generate
  num_generate = gen_size

  input_eval = [char_to_ind[s] for s in start_seed] # Vectorizing starting seed text

  input_eval = tf.expand_dims(input_eval,0) # Expand to match batch format shape

  text_generated = [] # Empty list to hold resulting generated text

  # Temperature effects randomness in our resulting text
  # The term is derived from entropy/thermodynamics.
  # The temperature is used to effect probability of next characters.
  # Higher temperature == less surprising/ more expected
  # Lower temperature == more surprising / less expected
 
  temperature = temp

  # Here batch size == 1
  model.reset_states()

  for i in range(num_generate):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions,0)
    predictions = predictions/temperature

    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

    input_eval = tf.expand_dims([predicted_id],0)

    text_generated.append(ind_to_char[predicted_id])

  return (start_seed+"".join(text_generated))
  
st.success(generate_text(model,character_name,words))