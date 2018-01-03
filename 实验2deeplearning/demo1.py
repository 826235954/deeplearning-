
import numpy as np
from keras.layers import Input  
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D, Dense, Activation
from keras.models import Sequential
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from keras.datasets import mnist


#*******************************data preprocessing*******************************************
#load data
img_rows, img_cols = 28, 28
input_shape = 28, 28,1
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.double(np.reshape(x_train, (len(x_train), img_rows , img_cols,1)))/255
x_test = np.double(np.reshape(x_test, (len(x_test), img_rows,img_cols,1)))/255

#*************************construct a convolutional stack autoencoder************************

auto_encoder = Sequential()
auto_encoder.add(Conv2D(5, (8, 8), activation='relu', padding='same',input_shape=input_shape))  
auto_encoder.add(MaxPooling2D((2, 2), padding='same'))  
auto_encoder.add(UpSampling2D((2, 2))) 
auto_encoder.add(Conv2D(5, (8, 8), activation='relu', padding='same')) 
auto_encoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same')) 

#**********************************model compile*********************************************

auto_encoder.compile(optimizer='sgd', loss='mean_squared_error')
auto_encoder.summary()

#****************************************training*******************************************
for q in range(5):
	auto_encoder.fit(x_train, x_train,  
				       epochs=10,
				       batch_size=128,
				       shuffle=True,
				       validation_data=(x_test, x_test)) 
			       
	#obtain the model predicetions

	decoded_imgs = auto_encoder.predict(x_test)  

	#save the model
	auto_encoder.save('denoise_epoch_5'+'.h5')

	plt.figure(figsize=(24, 4))
	n=12

	#show the differents between the groudtruth and model predictions

	d = {}

	for i in range(len(y_test)):
		a = y_test[i]
		d[a]=i
		if len(d)>10:
			break
	e = [2,0,1,7,2,1,4,3,0,0,6,4]

	for i in range(n):
				  # display original
							ax = plt.subplot(2, n, i + 1)
							plt.imshow(x_test[d[e[i]]].reshape(28, 28))
							plt.gray()
							ax.get_xaxis().set_visible(False)
							ax.get_yaxis().set_visible(False)
							# display reconstruction
							ax = plt.subplot(2, n, i + 1 + n)
							plt.imshow(decoded_imgs[d[e[i]]].reshape(28, 28))
							plt.gray()
							ax.get_xaxis().set_visible(False)
							ax.get_yaxis().set_visible(False)
				
	plt.show()



