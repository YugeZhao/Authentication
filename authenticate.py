import numpy
#print (numpy.__path__)
import numpy as np
import skimage as sk
import skimage.io as skio 
import scipy as sci
import matplotlib as mplt
import matplotlib.pyplot as plt
import sys
import cv2
from cv2 import circle
from cv2 import ml

##readImage laods the original image to greyscale image, decrease noise, and find edges
def readImage(imname, d, sigmaC, sigmaS):
    im_o = cv2.imread(imname)
    im = cv2.resize(im_o, (100, 100)) 
    #print(im.shape)
    image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image,d,sigmaC,sigmaS)
    image_edge = cv2.Canny(image,150,200)
    image_edge_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    #cv2.imshow('edges',image_edge_thresh)
    return image_edge_thresh

def detect_features(image):
    detector = cv2.SimpleBlobDetector()
    blobs = detector.detect(image)
    return blobs


def detect_contour(image):
    (im, contours, hier) = cv2.findContours(image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key = cv2.contourArea, reverse = True)
    cut_logos = None
    for contour in contours:
        epi = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.01*epi, True)
        if len(approximation) > 15:
            #print(len(approximation))
            (x,y),r = cv2.minEnclosingCircle(approximation)
            c = (int(x), int(y))
            cv2. circle(image,c,int(r),(0,0,255))
            
            cut_logos = approximation
    #print(len(cut_logos[0]))
    cv2.drawContours(image, cut_logos, 1, (0,0,255),1)
    cv2.imshow("logo", image)
    cv2.waitKey(0)
    return  cut_logos

training = []
testing = []
training_labels = []
testing_labels=[]

imname1 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake10.jpg'
d_fake, sigmaC_fake, sigmaS_fake = 12, 20, 20
fake1 = readImage(imname1, d_fake, sigmaC_fake, sigmaS_fake)
logo1 = detect_contour(fake1)
#print(logo1)
training.append(logo1)
training_labels.append(0)
#cv2.waitKey(0)
#cv2.destroyWindow('edges')

imname2 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake101.jpeg'
fake2 = readImage(imname2, d_fake, sigmaC_fake, sigmaS_fake)
logo2 = detect_contour(fake2)
training.append(logo2)
training_labels.append(0)
#cv2.waitKey(0)
#cv2.destroyWindow('edges')

imname3 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake11.jpeg'
fake3 = readImage(imname3, d_fake, sigmaC_fake, sigmaS_fake)
logo3 = detect_contour(fake3)
training.append(logo3)
training_labels.append(0)

imname4 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake11.jpg'
fake4 = readImage(imname4, d_fake, sigmaC_fake, sigmaS_fake)
logo4 = detect_contour(fake4)
training.append(logo4)
training_labels.append(0)

imname5 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake111.jpeg'
fake5 = readImage(imname5, d_fake, sigmaC_fake, sigmaS_fake)
logo5 = detect_contour(fake5)
training.append(logo5)
training_labels.append(0)

imname6 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake112.jpeg'
fake6 = readImage(imname6, d_fake, sigmaC_fake, sigmaS_fake)
logo6 = detect_contour(fake6)
training.append(logo6)
training_labels.append(0)

imname7 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake2.jpg'
fake7 = readImage(imname7, d_fake, sigmaC_fake, sigmaS_fake)
logo7 = detect_contour(fake7)
training.append(logo7)
training_labels.append(0)

#imname8 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake22.jpeg'
#fake8 = readImage(imname8, d_fake, sigmaC_fake, sigmaS_fake)
#logo8 = detect_contour(fake8)
#training.append(logo8)
#training_labels.append(0)

imname9 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake221.jpeg'
fake9 = readImage(imname9, d_fake, sigmaC_fake, sigmaS_fake)
logo9 = detect_contour(fake9)
training.append(logo9)
training_labels.append(0)

imname10 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake3.jpg'
fake10 = readImage(imname10, d_fake, sigmaC_fake, sigmaS_fake)
logo10 = detect_contour(fake10)
training.append(logo10)
training_labels.append(0)

imname11 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake31.jpg'
fake11 = readImage(imname11, d_fake, sigmaC_fake, sigmaS_fake)
logo11 = detect_contour(fake11)
training.append(logo11)
training_labels.append(0)

#imname12 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake6.jpg'
#fake12 = readImage(imname12, d_fake, sigmaC_fake, sigmaS_fake)
#logo12 = detect_contour(fake12)
#training.append(logo12)
#training_labels.append(0)

imname13 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake61.jpeg'
fake13 = readImage(imname13, d_fake, sigmaC_fake, sigmaS_fake)
logo13 = detect_contour(fake13)
training.append(logo13)
training_labels.append(0)

imname14 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake62.jpeg'
fake14 = readImage(imname14, d_fake, sigmaC_fake, sigmaS_fake)
logo14 = detect_contour(fake14)
training.append(logo14)
training_labels.append(0)

imname15 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake7.jpg'
fake15 = readImage(imname15, d_fake, sigmaC_fake, sigmaS_fake)
logo15 = detect_contour(fake15)
testing.append(logo15)
training_labels.append(0)

imname16 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake8.jpg'
fake16 = readImage(imname16, d_fake, sigmaC_fake, sigmaS_fake)
logo16 = detect_contour(fake16)
testing.append(logo16)
testing_labels.append(0)

imname17 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake81.jpg'
fake17 = readImage(imname17, d_fake, sigmaC_fake, sigmaS_fake)
logo17 = detect_contour(fake17)
testing.append(logo17)
testing_labels.append(0)

#imname18 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake82.jpeg'
#fake18 = readImage(imname18, d_fake, sigmaC_fake, sigmaS_fake)
#logo18 = detect_contour(fake18)
#testing.append(logo18)
#testing_labels.append(0)

imname19 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake9.jpg'
fake19 = readImage(imname19, d_fake, sigmaC_fake, sigmaS_fake)
logo19 = detect_contour(fake19)
testing.append(logo19)
testing_labels.append(0)

imname20 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake91.jpg'
fake20 = readImage(imname20, d_fake, sigmaC_fake, sigmaS_fake)
logo20 = detect_contour(fake20)
testing.append(logo20)
testing_labels.append(0)

imname21 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/fake92.jpeg'
fake21 = readImage(imname21, d_fake, sigmaC_fake, sigmaS_fake)
logo21 = detect_contour(fake21)
testing.append(logo21)
testing_labels.append(0)

imname22 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real1.png'
d_real, sigmaC_real, sigmaS_real = 5, 15, 15
real1 = readImage(imname22, d_real, sigmaC_real, sigmaS_real)
logo22 = detect_contour(real1)
training.append(logo22)
training_labels.append(1)

#imname23 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real11.png'
#real2 = readImage(imname23, d_real, sigmaC_real, sigmaS_real)
#logo22 = detect_contour(real2)
#training.append(logo22)
#training_labels.append(1)

imname24 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real12.png'
real3 = readImage(imname24, d_real, sigmaC_real, sigmaS_real)
logo23 = detect_contour(real3)
training.append(logo23)
training_labels.append(1)

imname25 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real2.png'
real4 = readImage(imname25, d_real, sigmaC_real, sigmaS_real)
logo24 = detect_contour(real4)
training.append(logo24)
training_labels.append(1)

imname26 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real21.png'
real5 = readImage(imname26, d_real, sigmaC_real, sigmaS_real)
logo25 = detect_contour(real5)
training.append(logo25)
training_labels.append(1)

#imname27 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real22.png'
#real6 = readImage(imname27, d_real, sigmaC_real, sigmaS_real)
#logo26 = detect_contour(real6)
#training.append(logo26)
#training_labels.append(1)

imname28 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real4.png'
real7 = readImage(imname28, d_real, sigmaC_real, sigmaS_real)
logo27 = detect_contour(real7)
training.append(logo27)
training_labels.append(1)

imname29 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real41.png'
real8 = readImage(imname29, d_real, sigmaC_real, sigmaS_real)
logo28 = detect_contour(real8)
training.append(logo28)
training_labels.append(1)

#imname30 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real42.png'
#real9 = readImage(imname30, d_real, sigmaC_real, sigmaS_real)
#logo29 = detect_contour(real9)
#training.append(logo29)
#training_labels.append(1)

imname31 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real7.png'
real10 = readImage(imname31, d_real, sigmaC_real, sigmaS_real)
logo30 = detect_contour(real10)
testing.append(logo30)
testing_labels.append(1)

#imname32 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real71.png'
#real11 = readImage(imname32, d_real, sigmaC_real, sigmaS_real)
#logo31 = detect_contour(real11)
#testing.append(logo31)
#testing_labels.append(1)

#imname33 = '/Users/annazhao/eclipse-workspace/Authentication Program/src/Images/real72.png'
#real12 = readImage(imname33, d_real, sigmaC_real, sigmaS_real)
#logo32 = detect_contour(real12)
#testing.append(logo32)
#testing_labels.append(1)
 
print(training)
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setDegree(0.0)
# svm.setGamma(0.0)
# svm.setCoef0(0.0)
# svm.setC(0)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
svm.train(training, cv2.ml.ROW_SAMPLE, training_labels)

sample_data = np.array(testing, np.float32)
result = svm.predict(sample_data)
print(result)

predict_real = np.count_nonzero(result)
accuracy = predict_real/result.size

print('accuracy is' + accuracy)








