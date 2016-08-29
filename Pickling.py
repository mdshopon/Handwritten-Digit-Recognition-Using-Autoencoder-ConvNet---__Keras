
from PIL import Image
from os import listdir
from os.path import isfile, join
import PIL.ImageOps
import matplotlib.cm as cm
import numpy as np
from skimage import color
from skimage import io
import pickle

# print (A.shape)
def makePickle():
     Name = []
     for filename in listdir("BengaliBMPConvertGray"):
          if filename.endswith(".bmp"):
               Name.append("BengaliBMPConvertGray/" + filename)
     # print(Name)
     x = np.array([np.array(Image.open(fname)) for fname in Name])
     filehandle = open('BengaliBMP.p', 'wb')
     pickle.dump(x, filehandle, protocol=2)

def rgbtogray():
     for filename in listdir("BengaliBMPConvert"):
          if filename.endswith(".bmp"):
                    FILE="BengaliBMPConvert/"+filename;
                    image = color.rgb2gray(io.imread(FILE))
                    FILE="BengaliBMPConvertGray/"+filename
                    io.imsave(FILE,image)
          else:
               continue

def invertBackground():

     for filename in listdir("BengaliBMPConvert"):
          if filename.endswith(".bmp"):
               #FILE="BengaliBMP/"+filename;
               image = Image.open(FILE)
               inverted_image = PIL.ImageOps.invert(image)
               if filename.endswith('.bmp'):
                    filename = filename[:-4]
               FILE2="BengaliBMPConvert/"+filename+".bmp"
               inverted_image.save(FILE2)
               print(FILE2)
          else:
               continue


makePickle()