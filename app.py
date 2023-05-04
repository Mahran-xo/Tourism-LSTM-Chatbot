import asyncio
import uvicorn
import json
from pydantic import BaseModel
from fastapi import FastAPI , File , UploadFile
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras import layers , activations , models , preprocessing


with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)


model = load_model('LSTM.h5')


def make_inference_models():
    
    # Encoder model
    encoder_inputs = tf.keras.layers.Input(shape=(None,))
    encoder_embeddings = model.layers[2](encoder_inputs)
    encoder_lstm = model.layers[4]
    _, state_h_enc, state_c_enc = encoder_lstm(encoder_embeddings)
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
    
    # Decoder model
    decoder_inputs = tf.keras.layers.Input(shape=(None,))
    decoder_embeddings = model.layers[3](decoder_inputs)
    decoder_lstm = model.layers[5]
    decoder_dense = model.layers[6]
    decoder_state_input_h = tf.keras.layers.Input(shape=(200,))
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_embeddings, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)
    
    return encoder_model, decoder_model



def str_to_tokens( sentence : str ):
    words = sentence.lower().split()
    tokens_list = list()
    for word in words:
        tokens_list.append( tokenizer.word_index[ word ] ) 
    return preprocessing.sequence.pad_sequences( [tokens_list] , maxlen=13 , padding='post')



def get_resp(text):
    enc_model , dec_model = make_inference_models()
    while True:
        try:
            user_input = text
            
            if len(user_input.split()) < 2:
                raise ValueError('Please enter a valid question.')
            elif not all(word in tokenizer.word_index for word in text.split()):
                raise ValueError("Please enter a question with words that are in the model's vocabulary.")
            else:
                user_input = re.sub(r'[^\w\s]', '', text)
                states_values = enc_model.predict(str_to_tokens(text), verbose=0)
                empty_target_seq = np.zeros((1, 1))
                empty_target_seq[0, 0] = tokenizer.word_index['start']
                stop_condition = False
                decoded_translation = ''
                while not stop_condition:
                    dec_outputs, h, c = dec_model.predict([empty_target_seq] + states_values, verbose=0)
                    sampled_word_index = np.argmax(dec_outputs[0, -1, :])
                    sampled_word = None
                    for word, index in tokenizer.word_index.items():
                        if sampled_word_index == index:
                            decoded_translation += ' {}'.format(word)
                            sampled_word = word

                    if sampled_word == 'end' or len(decoded_translation.split()) > 85:
                        stop_condition = True

                    empty_target_seq = np.zeros((1, 1))
                    empty_target_seq[0, 0] = sampled_word_index
                    states_values = [h, c]

        except ValueError as e:
            print('Error:', e)
            continue

        return decoded_translation[:-3]


app = FastAPI()

class RequestBody(BaseModel):
    text: str

@app.post('/')
def process_text(request_body: RequestBody):

    text = request_body.text

    processed_text = get_resp(text)

  
    return {'Response': processed_text}