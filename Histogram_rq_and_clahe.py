import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def histogramEqal():
    root=os.getcwd()
    imgPath=os.path.join(root,'histo_images/WP_Rend_02.png' )
    img=cv.imread(imgPath,cv.IMREAD_GRAYSCALE)
    hist=cv.calcHist([img],[0],None,[256],[0,256])
    cdf=hist.cumsum()
    cdfNorm=cdf*float(hist.max())/cdf.max()

    plt.figure()
    plt.subplot(231)
    plt.imshow(img,cmap='gray')
    plt.subplot(234)
    plt.plot(hist)
    plt.plot(cdfNorm,color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')

    equImg=cv.equalizeHist(img)
    equhist=cv.calcHist([equImg],[0],None,[256],[0,256])
    equcdf=equhist.cumsum()
    equcdfNorm=equcdf*float(hist.max())/equcdf.max()

    plt.subplot(232)
    plt.imshow(equImg, cmap='gray')
    plt.subplot(235)
    plt.plot(equhist)
    plt.plot(equcdfNorm, color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')

    claheObj=cv.createCLAHE(clipLimit=5,tileGridSize=(8,8))
    claheImg=claheObj.apply(img)
    clahehist = cv.calcHist([claheImg], [0], None, [256], [0, 256])
    clahecdf = clahehist.cumsum()
    clahecdfNorm = clahecdf * float(clahehist.max()) / clahecdf.max()

    plt.subplot(231)
    plt.title("Original Image")

    plt.subplot(232)
    plt.title("Histogram Equalized")

    plt.subplot(233)
    plt.title("CLAHE Enhanced")

    plt.subplot(234)
    plt.title("Original Histogram & CDF")

    plt.subplot(235)
    plt.title("Equalized Histogram & CDF")

    plt.subplot(236)
    plt.title("CLAHE Histogram & CDF")

    plt.subplot(233)
    plt.imshow(claheImg, cmap='gray')
    plt.subplot(236)
    plt.plot(clahehist)
    plt.plot(clahecdfNorm, color='b')
    plt.xlabel('pixel intensity')
    plt.ylabel('# of pixels')

    plt.show()
if __name__=='__main__':
    histogramEqal()
