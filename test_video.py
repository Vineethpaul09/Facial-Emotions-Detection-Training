import os  
import cv2  
import numpy as np  
from keras.models import model_from_json  
from keras.preprocessing import image  
import time
import csv
import os
import os.path
from collections import Counter
import matplotlib.pyplot as plt


# emotions array 
emotions_csv = []
# list of emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'] 

# funtion graph will append all the emotions
def graph(a):
    emotions_csv.append(a)
    print(emotions_csv)
 
#load model, which we have saved during the traing 
model = model_from_json(open("./model.json", "r").read())  
#load weights which we have saved during the traing 
model.load_weights('C:/Vineeth/Learning/Python - Advances/project final code/model_filter.h5')  
# Cascading classifiers are trained with several hundred positive sample views of a particular object it is imported from the opencv
face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')  
# video capture
cap=cv2.VideoCapture(0)  
  
while True:  
# captures frame and returns boolean value and captured image 
    ret,test_img=cap.read() 
    if not ret:  
        continue  
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) #an image from one color space to another, in this case BGR to gray color
    # face detection 
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)   
 
    for (x,y,w,h) in faces_detected:  
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7) # making an rectangle around the face 
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image  
        roi_gray=cv2.resize(roi_gray,(48,48))  # resizing roi 
        img_pixels = image.img_to_array(roi_gray)  
        img_pixels = np.expand_dims(img_pixels, axis = 0)  
        img_pixels /= 255  
        # making the prediction with the model.
        predictions = model.predict(img_pixels)  
        #find max indexed array  
        max_index = np.argmax(predictions[0])  
        # list of emotion 
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
        # predicted emotions 
        predicted_emotion = emotions[max_index]  
        # text on image with the predited emotions 
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        graph(predicted_emotion)
    

    resized_img = cv2.resize(test_img, (1000, 700))  
    # image show
    cv2.imshow('Facial emotion analysis ',resized_img)
    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed  
        break  
    
cap.release()  
# destroy all windows
cv2.destroyAllWindows  

# after emotions are detected by the videocamera, then displaying result of the overall emotions using pie chart 
time.sleep(5)
my_dic = dict(Counter(emotions_csv))
print(my_dic)
# colors used in the pie chart
colors = ( "#003f5c", "#374c80", "#7a5195",
          "#bc5090", "#ef5675", "#ff764a", "#ffa600")
# wp will help to different the sizes, the linewidth is 1 and color is black         
wp = { 'linewidth' : 1, 'edgecolor' : "black" }
# Data to plot
labels = [] # labels are emotions
sizes = [] # number of emotions 

for x, y in my_dic.items():
    labels.append(x)
    sizes.append(y)
    
def func(pct, allvalues):
    absolute = int(pct / 100.*np.sum(allvalues))
    return "{:.1f}%\n({:d} emotions)".format(pct, absolute)
# plotting the pie chart
fig, ax = plt.subplots(figsize =(10, 7)) # fig size
wedges, texts, autotexts = ax.pie(sizes,
                                  autopct = lambda pct: func(pct, sizes),
                                  labels = labels,
                                  shadow = False,
                                  colors = colors,
                                  startangle = 90,
                                  wedgeprops = wp,
                                  textprops = dict(color ="white"))

# For the legend               
for a in emotions:
     labels.append(a) if a not in labels else labels
     
n = len(labels) - len(sizes)
for a in range(0,n):
    sizes.append(0)
print(sizes) # 
ax.legend(wedges, labels,
          title ="Emotions",
          loc ="lower center",
          bbox_to_anchor =(1, 0, 0.5, 1))

plt.setp(autotexts, size = 8, weight ="bold")
ax.set_title("Detected Emotions") # title 

# show plot
plt.show()
