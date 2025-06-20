import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test) = load_data()

X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

img_shape = X_train.shape[1:]

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu', input_shape=img_shape))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(X_train,y_train,epochs=10,batch_size=128,validation_data=(X_test,y_test),verbose=1)
model.save("mnist_model.keras")

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
#plt.show()

predictions = model.predict(X_test)
pred_classes = np.argmax(predictions, axis=1)

plt.imshow(X_test[0].reshape(28,28), cmap='gray')
plt.title(f"Predicted: {pred_classes[0]}, True: {y_test[0]}")
#plt.show()


loss, acc = model.evaluate(X_test,y_test,verbose=0)
print(loss,acc*100)