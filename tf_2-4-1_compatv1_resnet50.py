import tensorflow as tf
assert(tf.__version__ == "2.4.1")
import keras
import numpy as np
import sys
import tflearn

_Dataset_Path = "train"

print("Using TF " + tf.__version__)
print("Using Python " + sys.version)

print("Loading dataset...")
x_train, y_train = tflearn.data_utils.image_preloader(_Dataset_Path, image_shape=(224, 224), grayscale=False, mode="folder", categorical_labels=True, normalize=True)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
print("Train Labels:")
print(y_train)

model = tf.compat.v1.keras.applications.resnet50.ResNet50(weights=None, include_top=True, classes=5)

optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", metrics=['accuracy','categorical_crossentropy'], optimizer=optimizer)
print("Maximum weight in TF 2.4.1 at layer [-6]:")
print(np.max(model.layers[-6].get_weights()[0]))
print("Minimum weight in TF 2.4.1 at layer [-6]:")
print(np.min(model.layers[-6].get_weights()[0]))

print("Using TF 2.4.1 initialized Model:")
print("Metrics of untrained model on train-set")
metrics = model.evaluate(x_train, y_train)
print(model.metrics_names)
print(metrics)

print("Predicted labels:")
print(model.predict(x_train))

print("Using TF 1.13.1 initialized Model:")
model = tf.compat.v1.keras.models.load_model("TF-1-13-1-initialWeights_resnet50.h5")
print("Metrics of untrained model on train-set")
metrics = model.evaluate(x_train, y_train)
print(model.metrics_names)
print(metrics)
print("Predicted labels:")
pred = model.predict(x_train)
print(pred)
print("Standalone CategoricalCrossEntropy")
cce = keras.losses.CategoricalCrossentropy()
print(cce(y_train, pred).eval(session=tf.compat.v1.Session()))
