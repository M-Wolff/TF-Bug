import numpy as np
import sys
from pathlib import Path
import pickle
import matplotlib.pyplot as plt


def load_history(path):
    history = None
    with open(path, "rb") as pickle_file:
        history = pickle.load(pickle_file)
    return history


loss_key_tr = "loss"
loss_key_te = "val_loss"
acc_key_tr = "acc"
acc_key_te = "val_acc"

fig, axs1 = plt.subplots(2,2)
axs2 = 2*[2*[None]]
axs2 = np.array(axs2)

axs2[0][0] = axs1[0][0].twinx()
axs2[0][1] = axs1[0][1].twinx()
axs2[1][0] = axs1[1][0].twinx()
axs2[1][1] = axs1[1][1].twinx()


resnet_old = load_history("historySave1.13.1_R.dat")
resnet_new = load_history("historySave2.4.1_R.dat")
inception_old = load_history("historySave1.13.1_I.dat")
inception_new = load_history("historySave2.4.1_I.dat")

for subplot_number, net_type, net_type_str in [(0, resnet_old, "Resnet old"), (1, inception_old, "Inception old"), (2, resnet_new, "Resnet new"), (3, inception_new, "Inception new")]:
    ax1 = axs1[subplot_number // 2][subplot_number % 2]
    ax2 = axs2[subplot_number // 2][subplot_number % 2]
    
    ax1.plot(range(net_type[loss_key_te].__len__()),net_type[loss_key_te], color="r", label="Test Loss")
    ax2.plot(range(net_type[acc_key_te].__len__()),net_type[acc_key_te], color="b", label="Test Acc")
    ax1.plot(range(net_type[loss_key_tr].__len__()),net_type[loss_key_tr], color="r", linestyle="dashed", label="Train Loss")
    ax2.plot(range(net_type[acc_key_tr].__len__()),net_type[acc_key_tr], color="b", linestyle="dashed", label="Train Acc")
    
    ax1.set_title(net_type_str)

leg1 = axs1[0][0].legend(loc="upper right", bbox_to_anchor=(1,1.2))
axs2[0][0].legend(loc="upper right", bbox_to_anchor=(0.8,1.2))
plt.show()