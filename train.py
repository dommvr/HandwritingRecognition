import numpy as np
import pandas as pd
import os

def get_data():
    current_directory = os.getcwd()
    file = os.path.join(current_directory, 'data', 'A_Z_Handwritten_Data.csv')
    train_df = pd.read_csv(file)
    train_df = train_df.sample(frac=1)
    train_df = train_df.reset_index(drop=True)
    labels = train_df.iloc[:, 0]
    labels = labels.to_numpy()
    labels = np.eye(26)[labels]
    images = train_df.drop(train_df.columns[0], axis=1)
    images = images.to_numpy()
    images = images / 255

    return labels, images

labels, images = get_data()

w_i_h = np.random.uniform(-0.5, 0.5, (128, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (26, 128))

b_i_h = np.zeros((128, 1))
b_h_o = np.zeros((26, 1))

learn_rate = 0.01
epochs = 3
nr_correct = 0

for epoch in range(epochs):
    for lable, image in zip(labels, images):
        lable.shape += (1,)
        image.shape += (1,)

        h_pre = b_i_h + w_i_h @ image
        h = 1 / (1 + np.exp(-h_pre))

        o_pre = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(-o_pre))

        e = 1 / len(o) * np.sum((o - lable) ** 2, axis=0)
        nr_correct += int(np.argmax(o) == np.argmax(lable))

        delta_o = o - lable
        w_h_o += -learn_rate * delta_o @ np.transpose(h)
        b_h_o += -learn_rate * delta_o

        delta_h = np.transpose(w_h_o) @ delta_o * (h * (1 - h))
        w_i_h += -learn_rate * delta_h @ np.transpose(image)
        b_i_h += -learn_rate * delta_h

    accuracy = round((nr_correct / images.shape[0]) * 100, 2)
    print(f"Epoch {epoch + 1} accuracy: {accuracy}%")
    nr_correct = 0

#Save trained neural network (save neural network weights)
weights = {
    'b_i_h': b_i_h,
    'w_i_h': w_i_h,
    'b_h_o': b_h_o,
    'w_h_o': w_h_o
}

current_directory = os.getcwd()
weights_folder = os.path.join(current_directory, 'trained_neural_network')
os.makedirs(weights_folder, exist_ok=True)

for i in weights:
    np.save(os.path.join(weights_folder, f"{i}.npy"), weights[i])