from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np
from PIL import Image

digits = datasets.load_digits()
features = digits.data
labels = digits.target
print(digits.data.shape)

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

#print(clf.predict([features[-1]]))
img = Image.open("test.jpg")
#img = img.convert('1')
img = img.resize((8, 8), Image.BILINEAR)
img = np.array(img)
#print(img)

#img = misc.imresize(img, (8,8))
#img = img.astype(digits.images.dtype)
#img = misc.bytescale(img, high = 16, low = 0)
##print(img)

x_test =[]

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3)
print(x_test)

print(clf.predict([x_test]))	
