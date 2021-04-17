# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
from os import listdir
from numpy import asarray
from numpy import save
import tensorflow as tf
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


def show_pics(animal='dog', no=5):
	# define location of dataset
	folder = 'dataset/train/'
	# plot first few images
	for i in range(no):
		# define subplot
		pyplot.subplot(330 + 1 + i)
		# define filename
		filename = folder + animal +'.' + str(i) + '.jpg'
		# load image pixels
		image = imread(filename)
		# plot raw pixel data
		pyplot.imshow(image)
	# show the figure
	pyplot.show()



show_pics('dog',7)

# define location of dataset
folder = 'dataset/train/'
photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0.0
	if file.startswith('cat'):
		output = 1.0
	# load image
	photo = load_img(folder + file, target_size=(200, 200))
	# convert to numpy array
	photo = img_to_array(photo)
	# store
	photos.append(photo)
	labels.append([photo,output])
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('dogs_vs_cats_photos.npy', photos)
save('dogs_vs_cats_labels.npy', labels)
print(labels)
		
