import PIL 
import numpy as np
from numpy import asarray 

print('Installed Pillow Version:', PIL.__version__)


from PIL import Image
"""
image = Image.open('/home/mnit/Music/ImageToNumpy/img.jpg')


# summarize some details about the image 
print(image.format) 
print(image.size) 
print(image.mode)


numpydata = asarray(image)

print(type(numpydata))

print(numpydata.shape)

#print(numpydata)



# method-2
np_img = np.array(image)
print(np_img.shape)


# getting back the imag from the converted array

pilImg = Image.fromarray(numpydata)
print(type(pilImg))

# Let us check  image details 
print(pilImg.mode) 
print(pilImg.size)

"""

# using open cv

import cv2 

image2 = cv2.imread('/home/mnit/Music/ImageToNumpy/img.jpg')

print(type(image2))


# BGR -> RGB 
img2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 

cv2.imwrite('opncv_img.jpg', img2) 
print (type(img2))

data = np.array(img2)
flattened = data.flatten()
print data.shape
print flattened.shape
print flattened

#cv2.imshow("IMAGE", img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

