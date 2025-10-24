from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm(input_timesteps, input_features, output_dim, neurons=512, dropout=0.4, stateful=False):
    model = Sequential()
    model.add(
        LSTM(
            neurons,
            batch_input_shape=(None, input_timesteps, input_features),
            stateful=stateful,
            dropout=dropout,
        )
    )
    model.add(Dense(output_dim))
    return model
