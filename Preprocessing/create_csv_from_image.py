import numpy as np
from PIL import Image
from os.path import isfile, join
from os import listdir

imagePath = "/home/manash/Desktop/Convolutional-Neural-Network-master/SampleDataset"
imageFiles = [imagePath + '/' + f for f in listdir(imagePath) if isfile(join(imagePath, f))]
imageFiles = sorted(imageFiles)
labels = [int(lbl[-5]) for lbl in imageFiles]

with file('features_2.csv', 'w') as outFile:
    for index, imageFile in enumerate(imageFiles):
        img = Image.open(imageFile)
        pixel_data = list(img.getdata())
        pixel_data.insert(0, labels[index])
        pixel_data_np = np.asarray([pixel_data])
        np.savetxt(outFile, pixel_data_np, delimiter=',')
