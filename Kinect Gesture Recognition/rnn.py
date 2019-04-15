import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, LSTM, Dense, concatenate, Bidirectional, SimpleRNN, Dropout

from normalize_frames import normalize_frames
from load_gestures import load_gestures

# 9. FORMAT DATA
gesture_sets = load_gestures()
gesture_sets = normalize_frames(gesture_sets, 36)

samples = []
labels = []

for gs in gesture_sets:
    for seq in gs.sequences:
        sample = np.vstack(list(map(lambda x: x.frame, seq.frames)))
        samples.append(sample)
        labels.append(gs.label)

X = np.array(samples)
Y = np.vstack(labels)

# Shuffle data
p = np.random.permutation(len(X))
X = X[p]
Y = Y[p]

# 10. CREATE AND TRAIN MODEL
batch_size = 24
epochs = 400
latent_dim = 24

input_layer = Input(shape=(X.shape[1:]))
lstm = LSTM(latent_dim)(input_layer)
lstm2 = LSTM(latent_dim,go_backwards = True)(input_layer)

bi_lstm = concatenate([lstm,lstm2])

#rnn = SimpleRNN(latent_dim + 1)(bi_lstm)

# dense1 = Dense(latent_dim, activation='relu')(bi_lstm)
# dense2 = Dense(latent_dim, activation='relu')(dense1)
# dense3 = Dropout(0.5)(dense2)
dense3 = Dense(latent_dim * 2, activation='relu')(bi_lstm)

pred = Dense(len(gesture_sets), activation='softmax')(dense3)

model = Model(inputs=input_layer, outputs=pred)
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["acc"])
model.summary()

model.fit(X,
          Y,
          epochs=epochs,
          batch_size=batch_size,
          verbose=1,
          validation_split=0.3,
          shuffle=True)
