import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

from sklearn.metrics import classification_report


data = pd.read_csv('cover_data.csv') #paste in url where you downloaded dataset from below link
#https://www.kaggle.com/datasets/uciml/forest-cover-type-dataset

features = data.iloc[:,0:54]
labels = data.iloc[:, -1]

print(data.info())
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.15)

ct = ColumnTransformer([('standardize', StandardScaler(), ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                                                           'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                                                           'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                                                           'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
                                                           'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4'])], remainder='passthrough')

x_train_scaled = ct.fit_transform(x_train)
x_test_scaled = ct.transform(x_test)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns)

le = LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

y_train = tf.keras.utils.to_categorical(y_train, dtype='int64')
y_test = tf.keras.utils.to_categorical(y_test, dtype='int64')

model = Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(x_train_scaled.shape[1],)))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])

with tf.device('/GPU:0'):
    history = model.fit(x_train_scaled, y_train, epochs=12, batch_size=64, verbose=1)


y_estimate = model.predict(x_test_scaled, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_estimate))

plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Covert type model accuracy')
plt.show()

results = {
    '1st run': 'loss: 0.5399 - accuracy: 0.7659',
    '2nd run': 'loss:0.6537 - accuracy: 0.7229 (batch_size=54, learning_rate0.02)',
    '3rd run': 'loss: 0.3851 - accuracy: 0.8394 (batch_size=54, learning_rate=0.005',
    '4th run': 'loss: 0.3692 - accuracy: 0.8466 (batch_size=64 learning_rate=0.005',
    '5th run': 'loss: 0.3142 - accuracy: 0.8680 (batch_size=64 learning_rate=0.003',
    '6th run': 'loss: 0.3557 - accuracy: 0.8493 (batch_size=64 learning_rate=0.003, first layer 256 removed)',
    '7th run': 'loss: 0.3562 - accuracy: 0.8490 (batch_size=81, learning_rate=0.0015, first layer 256 removed)',
    '8th run': 'loss: 0.3063 - accuracy: 0.8715 (batch_size=72, learning_rate=0.0025, first layer 128 )',
    '9th run': 'loss: 0.3077 - accuracy: 0.8704 (batch_size=96, learning_rate=0.002, first layer 128 )',
    '10th run': 'loss: 0.3133 - accuracy: 0.8688 (batch_size=72, learning_rate=0.0025, first layer 128, 6 hidden )',
    '11th run': 'loss: 0.3053 - accuracy: 0.8719 (batch_size=72, learning_rate=0.0025, first layer 256, 5 hidden )',
    '12th run': 'loss: 0.2702 - accuracy: 0.8866 (batch_size=72, learning_rate=0.0020, first layer 512, 6 hidden )',
    '13th run': 'loss: 0.2611 - accuracy: 0.8906 (batch_size=128, learning_rate=0.0020, first layer 512, 6 hidden )',
    '14th run': 'loss: 0.2565 - accuracy: 0.8928 (batch_size=64, learning_rate=0.0015, first layer 512, 6 hidden )',
    '15th run': 'loss: 0.2558 - accuracy: 0.8919 (batch_size=64, learning_rate=0.0010, first layer 512, 6 hidden )',
    '16th run': 'loss: 0.1274 - accuracy: 0.9482 (batch_size=64, learning_rate=0.001, first layer 1024, 4 hidden)'
}
