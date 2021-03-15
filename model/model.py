import math
import tensorflow as tf
import kerastuner as kt
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


dataframe = pd.read_csv('dielectron.csv')

# We can see that Q1 and Q2 are categorical because they represent the charge of the electron in question. All the
# other model-relevant variables are numeric because they are values that represent some quantity (energy, angular
# momentum, position in space), not type or rank. Thus, as we move on, we will transform Q1 and Q2 such that their
# categorical characteristics will be retained when taken as an input for the model and not used as a numeric value.
# We drop the Run and Event variables because we don't want our predictions to factor in which run they happened on
# and instead find patterns in the data concerning the electrons themselves.


# Alter dataframe slightly in order to work with model
dataframe = dataframe.drop(columns=['Run', 'Event'])
dataframe = dataframe.dropna()                          # Regression inputs cannot be NA
dataframe.loc[(dataframe.Q1 == -1), 'Q1'] = 0           # CategoryEncoding inputs must be non-negative
dataframe.loc[(dataframe.Q2 == -1), 'Q2'] = 0

# Split train, test, and validation dataframes
test = dataframe.sample(frac=0.15, random_state=1337)
train = dataframe.drop(test.index)
val = train.sample(frac=0.2, random_state=1337)
train = train.drop(val.index)

print(f"""Using {len(train)} samples for training, {len(val)} for validation, and {len(test)} for testing""")


# Conversion function for Pandas Dataframes
# Converts Pandas Dataframe to Tensorflow Dataset
def df_ds(df, label, batchsize):
    df_target = df.pop(label)
    ds = tf.data.Dataset.from_tensor_slices((df.to_dict('list'), df_target.values)).batch(batchsize)
    return ds


batch_size = 16
# Converting train, test, and validation dataframes to TF Dataset format
# Note: Each input in the model will correspond to a tensor in the train dataset
train_slices = df_ds(train, 'M', batch_size)
test_slices = df_ds(test, 'M', batch_size)
val_slices = df_ds(val, 'M', batch_size)

# Output to verify that our dataset consists of tensors with M as the target
for x, y in train_slices.take(1):
    print('Input:', x)
    print('Target:', y)


# Standardizer function for dataset features
# One-hot encodes categorical variables and normalizes numerical variables using Keras
# Preprocessing layers
@tf.autograph.experimental.do_not_convert
def standardizer(name, dataset):
    feature_ds = dataset.map(lambda x, y: x[name])
    if name in ['Q1', 'Q2']:
        encoder = preprocessing.CategoryEncoding(output_mode='binary')
        encoder.adapt(feature_ds)
        return encoder
    else:
        normalizer = preprocessing.Normalization()
        normalizer.adapt(feature_ds)
        return normalizer


# Defining model inputs and passing them through a standardization layer
inputs = {key: tf.keras.layers.Input(shape=(1,), name=key.strip()) for key in train.keys()}
inputs_en = []

for i in inputs.keys():
    standardize = standardizer(i, train_slices)
    input_en = standardize(inputs[i])
    inputs_en.append(input_en)


# Neural network initializer
# Input parameter: number of units in the first Dense layer
# Note: A Sequential model can be used here but a Functional one is used to ease
# potential additions to inputs in the future
def build_model(l1_units):
    x = layers.concatenate(inputs_en)
    x = layers.Dense(l1_units, 'relu')(x)
    x = layers.Dense(128, 'relu')(x)
    x = layers.Dense(64, 'relu')(x)
    outputs = layers.Dense(1, 'linear')(x)
    network = keras.Model(inputs, outputs)

    network.compile(optimizer=keras.optimizers.Adam(),
                    loss=keras.losses.MAE,
                    metrics=['mse'])

    return network


# Function to plot model MSE and Loss
def plot_model(model_history):
    plt.plot(model_history.history['mse'])
    plt.plot(model_history.history['val_mse'])
    plt.title('Model MSE Rate')
    plt.ylabel('MSE')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()


# Initializing model with 256 units in the first Dense layer. With this choice, the model features a common
# neural network architecture with the most units being in the first layer and decreasing until reaching the
# final number of outputs required by the model
model = build_model(256)
print(model.summary())
keras.utils.plot_model(model, 'model-architecture.png', rankdir='LR', show_shapes=True)

history = model.fit(train_slices, epochs=100, validation_data=val_slices)

# Evaluating model performance on test dataset
eval_result1 = model.evaluate(test_slices)
print(f"""Model RMSE (evaluated on test set): {math.sqrt(eval_result1[1])}""")                  # 0.498478063290

tf.keras.models.save_model(model, 'cern-ec-model')

plot_model(history)


# Keras Tuner model initializer
def build_tuner_model(hp):
    hp_units_1 = hp.Int('units', min_value=64, max_value=512, step=32)

    hp_x = layers.concatenate(inputs_en)
    hp_x = layers.Dense(hp_units_1, 'relu')(hp_x)
    hp_x = layers.Dense(128, 'relu')(hp_x)
    hp_x = layers.Dense(64, 'relu')(hp_x)
    hp_outputs = layers.Dense(1, 'linear')(hp_x)
    hp_model = keras.Model(inputs, hp_outputs)

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    hp_model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                     loss=keras.losses.MAE,
                     metrics=['mse'])

    return hp_model


# Tuner is configured to use HyperBand search (RandomSearch, Bayesian Optimization, etc are available)
tuner = kt.Hyperband(build_tuner_model, objective='val_mse', max_epochs=30, factor=3,
                     project_name='cern-ec-tuner-results')

# Tuner searches for best hyperparameter configuration over 50 epochs
tuner.search(train_slices, epochs=30, validation_data=val_slices)
best_hps = tuner.get_best_hyperparameters(1)[0]

# Initializing model using best hyperparameters from search
tuned_model = build_tuner_model(best_hps)
print(tuned_model.summary())
keras.utils.plot_model(tuned_model, 'tuned-model-architecture.png', rankdir='LR', show_shapes=True)

tuned_model_history = tuned_model.fit(train_slices, epochs=100, validation_data=val_slices)

tuned_eval_result = tuned_model.evaluate(test_slices)
print(f"""Tuned model RMSE (evaluated on test set): {math.sqrt(tuned_eval_result[1])}""")       # 0.710832147861

tf.keras.models.save_model(tuned_model, 'cern-ec-tuned-model')

plot_model(tuned_model_history)

# Somehow, we seem to have achieved a better RMSE from our default model so that's the one we choose for optional
# further training on the entire dataset and potential application
