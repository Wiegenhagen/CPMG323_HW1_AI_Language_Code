#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import gradio as gr

from tensorflow.keras.layers import TextVectorization


import os
import pandas as pd
import tensorflow as tf
import numpy as np


df = pd.read_csv('/Users/Arina/Documents/project_backend/train.csv')


X = df['comment_text']



MAX_FEATURES = 200000 



vectorizer = TextVectorization(max_tokens=MAX_FEATURES,
                               output_sequence_length=1800,
                               output_mode='int')



vectorizer.adapt(X.values)



vectorized_text = vectorizer(X.values)


model = tf.keras.models.load_model('toxicity.h5')


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    
    return text




interface = gr.Interface(fn=score_comment, 
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')



interface.launch(share=True)



