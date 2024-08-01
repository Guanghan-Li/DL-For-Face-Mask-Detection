import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

with_mask_path = r'/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/with_mask'
with_mask_files = os.listdir(with_mask_path)


without_mask_path = r'/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/without_mask'
without_mask_files = os.listdir(without_mask_path)

with_mask_labels = [1]*3725
without_mask_labels = [0]*3828

labels = with_mask_labels + without_mask_labels

#display image with/without mask
'''
img_mask = mpimg.imread('/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/with_mask/with_mask_1.jpg')
img_no = mpimg.imread('/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/without_mask/without_mask_1.jpg')
imgplot = plt.imshow(img_mask)
plt.show()
imgplot = plt.imshow(img_no)
plt.show()
'''
#image processing
data = []

with_mask_path = '/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/with_mask/'
for img_file in with_mask_files:
    image = Image.open(with_mask_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

without_mask_path = '/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/without_mask/'
for img_file in without_mask_files:
    image = Image.open(without_mask_path + img_file)
    image = image.resize((128,128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

#convert everything into a numpy arrays
x = np.array(data)
y = np.array(labels)

#train test split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

#scale the data make the rgb data from 0-1
x_trained_scaled = x_train/255
x_test_scaled = x_test/255

#build and use an CNN
import tensorflow as tf
#from tensorflow import keras

num_of_classes = 2
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, kernel_size = (3,3), activation = 'relu', input_shape=(128, 128, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(64, kernel_size = (3,3), activation = 'relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64,activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(num_of_classes, activation='sigmoid'))

#compile
model.compile(optimizer = 'adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

#train
history = model.fit(x_trained_scaled, y_train, validation_split=0.1, epochs=5)

#model Eval
loss,accuracy = model.evaluate(x_test_scaled, y_test)
print('Test accuracy = ', accuracy)

h= history
plt.plot(h.history['loss'], label = 'train loss')
plt.plot(h.history['val_loss'], label= 'validation loss')
plt.legend()
plt.show()

plt.plot(h.history['acc'], label = 'train acc')
plt.plot(h.history['val_acc'], label= 'validation acc')
plt.legend()
plt.show()


#predictive system
input_image_path = input("Path of the image to be predicted: ")
input_image = Image.open(input_image_path)
input_image_resized = input_image.resize((128,128))
input_image_resized = np.array(input_image_resized)
input_image_scaled = input_image_resized/255

input_image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])
input_prediction = model.predict(input_image_reshaped)
print(input_prediction)

input_pred_label = np.argmax(input_prediction)

print(input_pred_label)

if input_pred_label ==1:
    print("Wearing mask")
else:
    print("No mask")