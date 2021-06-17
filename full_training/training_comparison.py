import tensorflow as tf
import keras
import numpy as np
import sys
import tflearn
from pathlib import Path
import pickle


init_learning_rate = 0.001

def get_learning_rate(epoch):
    lr = init_learning_rate

    if epoch > 200:
        lr = 0.0005 * init_learning_rate
    elif epoch > 150:
        lr = 0.001 * init_learning_rate
    elif epoch > 100:
        lr = 0.01 * init_learning_rate
    elif epoch > 50:
        lr = 0.1 * init_learning_rate
    print("Epoch %s -> Learning-Rate: %s" % (epoch, lr))
    return lr

def evaluate_model(model, x, y_true):
    loss_acc = model.evaluate(x, y_true)
    print(model.metrics_names)
    print(loss_acc)
    acc = loss_acc[1] * 100
    loss = loss_acc[0]
    output_prediction = model.predict(x)
    print("prediction:")
    print(output_prediction)
    predicted_classes = np.argmax(output_prediction, axis=1)
    print("Predicted Classes")
    print(predicted_classes)
    y_true_classes = np.argmax(y_true, axis=1)
    print("True classes")
    print(y_true_classes)
    return acc, loss


_Dataset_Path = Path("/scratch/tmp/m_wolf37/Bachelorarbeit/datasets_exps/swedishLeaves3folds5/exps_ts10/")

net_type = "R" if sys.argv[1].lower() in ["resnet","r"] else "I"
img_size = 299 if net_type == "I" else 224
_BATCH_SIZE = 16


print("TF Version: " + tf.__version__)
print("Python Version " + sys.version)

print("Loading dataset...")
x_train, y_train = tflearn.data_utils.image_preloader(_Dataset_Path / "exp1" / "train", image_shape=(img_size, img_size), grayscale=False, mode="folder", categorical_labels=True, normalize=True)
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test, y_test = tflearn.data_utils.image_preloader(_Dataset_Path / "exp1" / "test", image_shape=(img_size, img_size), grayscale=False, mode="folder", categorical_labels=True, normalize=True)
x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


optimizer = keras.optimizers.Adam(lr=get_learning_rate(0))

if net_type == "R":
    model = keras.applications.resnet50.ResNet50(weights=None, include_top=True, classes=5)
elif net_type == "I":
    model = keras.applications.inception_v3.InceptionV3(weights=None, include_top=True, classes=5)


model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', "categorical_crossentropy"])
callbacks = [keras.callbacks.LearningRateScheduler(get_learning_rate)]

history = model.fit(x=x_train, y=y_train, validation_data=(x_test, y_test), epochs=200, shuffle=True, verbose=1, callbacks=callbacks, batch_size=_BATCH_SIZE)
with open("historySave" + tf.__version__ + "_" + net_type + ".dat", 'wb') as pickle_file:
        pickle.dump(history.history, pickle_file)

acc_test, loss_test = evaluate_model(model, x_test, y_test)
acc_train, loss_train = evaluate_model(model, x_train, y_train)
with open("results.txt", 'a+') as results:
        results.write(tf.__version__ + ";" + net_type + ";" + str(acc_train) + ";" + str(acc_test) + ";" + str(loss_test) + ";" + str(loss_train))
print("finished")