import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import string

def get_date():
    current_directory = os.getcwd()
    file = os.path.join(current_directory, 'data', 'A_Z_Handwritten_Data.csv')
    test_df = pd.read_csv(file)
    images = test_df.drop(test_df.columns[0], axis=1)
    images = images.to_numpy()
    images = images / 255

    return images

def number_to_alpha(number):
    number += 1
    num_to_alpha = dict(zip(range(1, 27), string.ascii_lowercase))

    return num_to_alpha[number]

images = get_date()

current_directory = os.getcwd()
weights_folder = os.path.join(current_directory, 'trained_neural_network')

b_i_h = np.load(os.path.join(weights_folder, 'b_i_h.npy'))
w_i_h = np.load(os.path.join(weights_folder, 'w_i_h.npy'))
b_h_o = np.load(os.path.join(weights_folder, 'b_h_o.npy'))
w_h_o = np.load(os.path.join(weights_folder, 'w_h_o.npy'))

while True:
    index = int(input('Index: '))
    if index == 666:
        exit()
    image = images[index]
    plt.imshow(image.reshape(28, 28), cmap='Greys')
    image.shape += (1,)

    h_pre = b_i_h + w_i_h @ image.reshape(784, 1)
    h = 1 / (1 + np.exp(-h_pre))

    o_pre = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(-o_pre))

    plt.title(f"This is a {number_to_alpha(o.argmax())}")
    plt.show()