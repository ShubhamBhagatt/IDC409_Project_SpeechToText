#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install jiwer')


# In[ ]:


get_ipython().system('pip install')


# In[ ]:


##importing the required libraries
import pandas as pd
import numpy as np
get_ipython().system('pip install tensorflow')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from IPython import display
from tensorflow.keras.optimizers import Adam
from jiwer import wer


# In[ ]:


data_url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
## extracting the data of LJspeech dataset using the keras in tensorflow
## untar = True is used to extract the file if it is archived
data_path = keras.utils.get_file("LJSpeech-1.1", data_url, untar =True)


# In[ ]:


pwd


# In[ ]:


wavs_path = data_path + "/LJSpeech-1.1/wavs/"
metadata_path = data_path + "/LJSpeech-1.1/metadata.csv"
##converting the csv file to a dataframe using pandas
metadata_df = pd.read_csv(metadata_path, sep = "|", header = None, quoting = 3)
metadata_df.head(10)


# In[ ]:


metadata_df.columns = ["file name", "transcription", "normalized transcription"]
##reshuffling the rows of the metadata df in a random order and not not

metadata_df = metadata_df.sample(frac = 1).reset_index (drop = True)
metadata_df.head(3)


# In[ ]:


## splitting the dataframe into two parts : training (90%), test (10%); using int to get an index at which to split
split = int(len(metadata_df) *0.0010)
df_train = metadata_df [:split]
df_test = metadata_df[split:]
print ("size of the training dataframe : ",{len(df_train)})
print ("size of the test dataframe : ",{len(df_test)})


# In[ ]:


#defining a list of allowed vocabulary
characters = [x for x in "abcdefghijklmnopqrstuvwxyz'?!"]
#converting the charactrs to integer values using keras; any character not in the list is given an empty string
char_to_num = keras.layers.StringLookup(vocabulary = characters, oov_token = "")
#converting the integer back to the character using keras, specifying it using invert
num_to_char = keras.layers.StringLookup(vocabulary = char_to_num.get_vocabulary(), oov_token = "", invert = True)
print (char_to_num.get_vocabulary())
print(char_to_num.vocabulary_size())


# In[ ]:


# setting the frame length, frame step and forward fourier transform rate
frame_length = 256
frame_step = 160
fft_length = 384
#defining function for getting a spectrogram of the audio files and its label
@tf.autograph.experimental.do_not_convert
def encode_single_sample (wav_file, label):
  #reading the audio file
  file = tf.io.read_file (wavs_path + wav_file + ".wav")
  #decoding the audio file using audio.decode_wav from the TensorFlow package to a float tensor
  audio,_ = tf.audio.decode_wav(file)
  #removing dimensions of size 1 from the tensor
  audio = tf.squeeze(audio, axis = -1)
  #changing the data type
  audio = tf.cast(audio, tf.float32)
  #converting the audio signal into a time frequency representation using Short Time Fourier Transfrom
  spectrogram = tf.signal.stft (audio, frame_length = frame_length, frame_step = frame_step, fft_length = fft_length)
  #calculating the magnitude of the spectrogram
  spectrogram = tf.abs(spectrogram)
  #normalizing the power
  spectrogram = tf.math.pow(spectrogram, 0.5)
  #performing the standardization of the spectrogram
  means = tf.math.reduce_mean(spectrogram, 1, keepdims=True)
  stddev = tf.math.reduce_std(spectrogram, 1, keepdims=True)
  spectrogram = (spectrogram - means)/(stddev - 1e-10)
  #splitting the label character and converting it to a numerical representation
  label = tf.strings.lower(label)
  label = tf.strings.unicode_split(label, input_encoding = "UTF-8")
  label = char_to_num(label)
  return spectrogram, label
batch_size = 32
file_names = np.array(df_train["file name"])
transcriptions = np.array(df_train["normalized transcription"])
train_dataset = tf.data.Dataset.from_tensor_slices(( file_names,transcriptions))
train_dataset = (train_dataset.map(encode_single_sample, num_parallel_calls = tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE))
test_dataset = tf.data.Dataset.from_tensor_slices((file_names, transcriptions))
test_dataset = (test_dataset.map(encode_single_sample, num_parallel_calls = tf.data.AUTOTUNE).padded_batch(batch_size).prefetch(buffer_size = tf.data.AUTOTUNE))


# In[ ]:


#defining function for CTCloss; y_true--> target , y_pred--> predicted
def CTCloss (y_true, y_pred):
  #getting the shape of the tensor and extracting the first dimension
  batch_len = tf.cast(tf.shape(y_true)[0], dtype = "int64")
  #calculating the second dimension
  input_length = tf.cast(tf.shape(y_pred)[1], dtype = "int64")
  #calculating the length of the label sequence
  label_length = tf.cast(tf.shape(y_true)[1], dtype = "int64")
  #giving it shape (batch_len, 1), creating a 2D tensor with ones
  input_length = input_length * tf.ones(shape = (batch_len, 1), dtype = "int64")
  label_length = label_length * tf.ones(shape = (batch_len, 1), dtype = "int64")
  #calculating the CTC loss
  #keras is deeplearning framework(API) in the TensorFlow package
  #the backend function does the mathematical operations on the tensors
  loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
  return loss


# In[ ]:


##defining a neural network using keras
##inputdim = dimension of the input
##outputdim = dimension of the input
##rnn layers = number of recurrent layers in the network
##rnn units = number of neurons
def draft_model (inputdim, outputdim, rnn_layers = 5, rnn_units = 128):
    #allowing the input to have variable length 
  input_spectrogram = layers.Input ((None, inputdim), name = "input")
#reshaping 
  x = layers.Reshape((-1, inputdim, 1), name = "expanddim")(input_spectrogram)
    #adding a convolutional layer
  x = layers.Conv2D( filters = 32, kernel_size = [11,41], strides = [2,2], padding = "same", use_bias = False, name = "conv_1",)(x)
#normalizing the layers 
  x = layers.BatchNormalization(name = "conv_1_bn")(x)
    #adding non-linearity using ReLU (Rectified Linear Unit)
  x = layers.ReLU(name = "conv_1_relu")(x)
#adding convolutional layer
  x = layers.Conv2D(filters = 32, kernel_size =[11,21], strides = [1,2], padding = "same", use_bias = False, name = "conv_2")(x)
  x = layers.BatchNormalization(name ="conv_2_bn")(x)
  x = layers.ReLU(name = "conv_2_relu")(x)
    #falttening the tensor into a 2D tensor
  x = layers.Reshape((-1, x.shape[-2]*x.shape[-1]))(x)
#Building a stack of birectional GRU (gated recurrent unit) layers  
  for i in range (1, rnn_layers + 1):
    recurrent = layers.GRU(units= rnn_units, activation = "tanh", recurrent_activation = "sigmoid", use_bias = True, return_sequences = True,
                           reset_after = True, name = f"gru_{i}",)
    x = layers.Bidirectional(recurrent, name = f"bidirectional_{i}", merge_mode="concat")(x)
    if i < rnn_layers:
      x = layers.Dropout(rate= 0.5)(x)
    #adding a dense layer to output
  x = layers.Dense(units = rnn_units * 2, name = "dense_1")(x)
  x = layers.ReLU(name = "dense_1_relu")(x)
    #preventing overfitting and adding noise 
  x = layers.Dropout ( rate = 0.5 )(x)
  output = layers.Dense(units = outputdim + 1, activation = "softmax")(x)
    #creating a model and naming it 
  model = keras.Model(input_spectrogram, output, name = "STT_Model")
#adding an optimizer, Adam, for training of model
  opt = Adam(learning_rate= 1e-4) 
  model.compile(optimizer = opt, loss= CTCloss)
  return model
fft_length = 384
#creating an instance of the model
model = draft_model(inputdim = fft_length // 2 + 1, outputdim= char_to_num.vocabulary_size(), rnn_units = 512,)
model.summary(line_length = 110)


# In[ ]:


#decoding the predictions made by the model 
def decode_batch_predictions (pred):
    #calculating the length of input sequence in a sample
  input_len = np.ones(pred.shape[0]) * pred.shape [1]
#decoding the CTC predictions using keras ctc_decode backend function; using the greedy decoder
  results = keras.backend.ctc_decode(pred, input_length = input_len, greedy = True)[0][0]
    #initiallizing list for the output
  output_text =[]
  for result in results :
        #converting to a NumPy array and then decoding as a UTF-8 string
    result = tf.strings.reduce_join(num_to_char(result)).numpy().decode("utf-8")
    output_text.append(result)
  return output_text
#defining a custom callback class to call in at a specific point in the model training to perfom an action 
class CallbackEval(keras.callbacks.Callback):
#initializing the method for the class; self --> instance of the class 
  def __init__(self, dataset):
        #calling the Kears callback function using the super function
    super().__init__()
    self.dataset = dataset
    #defining a specific callback method using at the end of each epoch 
  def on_epoch_end(self, epoch: int , logs = None):
     predictions = []
     targets = []
     for batch in self.dataset:
            #unloading the input (X) and the target (y), into batch 
      X, y = batch
    #using the predict function to get the precditions from the model 
      batch_predictions = model.predict(X)
        #decoding the predictions 
      batch_predictions = decode_batch_predictions (batch_predictions)
    #appending the predictions to the list 
      predictions.extend(batch_predictions)
      for label in y :
       label = (tf.strings.reduce_join (num_to_char(label)).numpy().decode("utf-8"))
       targets.append (label)
        #using the wer function in jiwer to get the word error rate 
     wer_score = wer (targets, predictions)
     print ("." *100)
     print (f"Word error rate : { wer_score: .4f}")
     print ("."*100)
   #printing two of the predictions at random
     for i in np.random.randint (0, len(predictions),2):
      print ("target : ", (targets[i]))
      print ("prediction: ", (predictions[i]))
      print ("." *100)


# In[ ]:


#specifying the number of epochs for training 
epochs = 1
#calling the custom Callbackeval function 
validation_callback = CallbackEval(test_dataset)
#training the model using the fit function in keras
history = model.fit(train_dataset, validation_data= test_dataset, epochs= epochs, callbacks = [validation_callback],)


# In[ ]:


#checking for the predictions in the test_dataset 
predictions = []
targets = []
for batch in test_dataset :
  X, y = batch
  batch_predictions = model.predict(X)
  batch_predictions = decode_batch_predictions (batch_predictions)
  predictions.extend(batch_predictions)
  for label in y :
       label = (tf.strings.reduce_join (num_to_char(label)).numpy().decode("utf-8"))
       targets.append (label)
wer_score = wer (targets, predictions)
print ("." *100)
print (f"Word error rate : { wer_score: .4f}")
print ("."*100)
print ("." *100)
# Check if the loop is being executed
print("Printing random samples:")
for i in range(2):
    random_index = np.random.randint(0, len(predictions))
    print("Random Index:", random_index)
    print("target : ", (targets[random_index]))
    print("prediction: ", (predictions[random_index]))
    print("." * 100)
for i in np.random.randint (0, len(predictions),2):
      print (f"target : {targets[i]}")
      print (f"prediction:  {predictions[i]}")
      print ("." *100)


# In[ ]:


#saving the model and its assets
model.save("stt.keras")


# In[ ]:


pwd

