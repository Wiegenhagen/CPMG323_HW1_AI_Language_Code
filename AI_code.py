# Import necessary libraries from Python/pip 
import os
import pandas as pd
import tensorflow as tf
import numpy as np


# Open file csv with training data, consisting out of 
# cloumns with comments and labels containg a binary value 
# if that comment falls under one of that lables 
# 1-true and 0-false 

df = pd.read_csv('/Users/Arina/Documents/project_backend/train.csv')

#Import text vectorization/tokens from tensorflow library 
#A preprocessing layer which maps text features to integer sequences 
from tensorflow.keras.layers import TextVectorization

# Split data into comments and the lables and add it to an array.
X = df['comment_text']
y = df[df.columns[2:]].values

# Number of words in the vocabulary 
MAX_FEATURES = 200000  #this can change 


# Initialize the vectorizer layer. 
vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800, # max length of a sentence (comment), if the lenght is bigger it means bigger model
                               output_mode='int') # outputs the words as int, and maps it to an int value. example Love= 1, hate = 4, shoes =400


# Train the vectorizer our words in our vocabulary/ comments . adapt method.  X.values = numpy.dsarray 
vectorizer.adapt(X.values)

#tokenising every saving it 
vectorized_text = vectorizer(X.values)


# Create our dataset (tensoflow data pipeline),each line is n data preprocessing step. 
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y)) # pass the data and lables 
dataset = dataset.cache()
dataset = dataset.shuffle(160000) # amount of samples / buffer size 
dataset = dataset.batch(16) # each batch with 16 samples 
dataset = dataset.prefetch(8)  # prevents bottlenecks 


#Take a part of the data out and assign it to a var  (9972 total batches of data)
train = dataset.take(int(len(dataset)*.7)) # training partition is 70% of all our data ( 6981 batches)
val = dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2)) # skip the 70% and take 20% of the data for the validation partition ( 1994 batches)
test = dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))# 10 % for test partition ( 997 batches)


# Deep learning part 
#Import sequentail API (fast and easy)
from tensorflow.keras.models import Sequential
#Layers to build the network 
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional, Dense, Embedding


# Model 
model = Sequential() # add the API to the model var
# Create the embedding layer 
model.add(Embedding(MAX_FEATURES+1, 32)) # number of words +1 for unkown words. Embeddding is 32 of length
# Bidirectional LSTM Layer
model.add(Bidirectional(LSTM(32, activation='tanh'))) # tanh CPU usages, specify by tensorflow
# Feature extractor Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
# Final layer 
model.add(Dense(6, activation='sigmoid')) # Maps to the number of different outputs that we got inside the neural network 
# 6 will allow us to output the same type of output as our labels, thus the outputs will have a value between 0 and 1  


model.compile(loss='BinaryCrossentropy', optimizer='Adam')

# train the model for 3 epochs (cycles around 1 hour each cycle on my computer)
history = model.fit(train, epochs=3, validation_data=val) # validation dataset is also pass through with train dataset


# train the model for 5 epochs (cycles around 1 hour each cycle on my computer)
history = model.fit(train, epochs=5, validation_data=val)


# Import mathplot to be able to present the validation loss and loss on a graph 
from matplotlib import pyplot as plt
plt.figure(figsize=(8,5))
pd.DataFrame(history.history).plot()
plt.show()



#Test predictions //code removed because not important to the model 




# Evaluate the model step 
# Import metrics 

from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy


# Metrics - Precision, recall and accuracy  (for false-pos, true-pos, false-neg, false-pos)
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()


# Perform the evaluation on all our datasets  
for batch in test.as_numpy_iterator(): 
    # Unpack the batch 
    X_true, y_true = batch
    # Make a prediction 
    yhat = model.predict(X_true)
    
    # Flatten the predictions (One big vector)
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)


# Print our predictions 
print(f'Precision: {pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{acc.result().numpy()}')


# Use the model 
# Import gradio to use as front end to test the model (anyone can use it when the link is active) 

import tensorflow as tf
import gradio as gr


# Save the model 
model.save('toxicity.h5')


# For loading the model 
model = tf.keras.models.load_model('toxicity.h5')


# Code to be able to pass through an input to test it against the model 

def score_comment(comment): #get input 
    vectorized_comment = vectorizer([comment]) #vectorize the input comment/words 
    results = model.predict(vectorized_comment) # pass it through the model 
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text #return the output in the format specifed above (default)


# Load the front end interface and how it will output  
interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)
