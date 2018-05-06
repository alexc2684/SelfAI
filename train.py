from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np

AI_NAME = 'Alex Chan'
E_IN_DIR = 'encoder_input.npy'
D_IN_DIR = 'decoder_input.npy'
VOCAB_SIZE = 6000
LATENT_DIM = 256
BATCH_SIZE = 64
EPOCHS = 5
MAX_Q = 30

#load data
encoder_in = np.load(E_IN_DIR)
decoder_in = np.load(D_IN_DIR)
NUM_SAMPLES = len(encoder_in)
decoder_out = np.zeros([NUM_SAMPLES, MAX_Q, VOCAB_SIZE], dtype=np.int32)
decoder_out[0:decoder_out.shape[0]-1] = decoder_in[1:]

#define encoder
encoder_inputs = Input(shape=(None,))
x = Embedding(VOCAB_SIZE, LATENT_DIM)(encoder_inputs)
x, hidden_h, hidden_c = LSTM(LATENT_DIM, return_state=True)(x)
encoder_states = [hidden_h, hidden_c]

#define decoder
decoder_inputs = Input(shape=(None,))
x = Embedding(VOCAB_SIZE, LATENT_DIM)(decoder_inputs)
x, _, _ = LSTM(LATENT_DIM, return_state=True)(x, initial_state=encoder_states)
decoder_outputs = Dense(VOCAB_SIZE, activation='softmax')(x)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_in, decoder_in], decoder_out, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=.2)
model.save("model_0.h5")
