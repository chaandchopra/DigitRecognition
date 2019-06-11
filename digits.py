from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import numpy as np

digits = datasets.load_digits()
features = digits.data
labels = digits.target
#print(len(labels))

clf = SVC(gamma = 0.001)
clf.fit(features, labels)

#print(clf.predict([features[-1]]))
img = misc.imread("test.jpg")
img = misc.imresize(img, (8,8))
img = img.astype(digits.images.dtype)
img = misc.bytescale(img, high = 16, low = 0)
#print(img)

x_test =[]

for eachRow in img:
	for eachPixel in eachRow:
		x_test.append(sum(eachPixel)/3)
#print(x_test)

print(clf.predict([x_test]))	
