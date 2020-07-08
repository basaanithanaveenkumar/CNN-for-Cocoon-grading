from keras.models import load_model
import cv2
from matplotlib import pyplot as plt

import numpy as np

model = load_model('vgg1.h5')

model.compile(loss='binary_crossentropy',

              optimizer='rmsprop',

              metrics=['accuracy'])

img = cv2.imread('test_g1.jpg')
dis= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.resize(img,(224,224))
dis= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = np.reshape(img,[1,224,224,3])

classes = model.predict_classes(img)
plt.imshow(dis)
plt.title('input image given')
plt.show()
if(classes==0):
    print("the input imsage is :defective cocoon")
else:
    print("the input image is :good cocooon")
