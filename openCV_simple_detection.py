''' 
PROGRAM DEVELOPPED BY THE SORBONNE STUDENT ASSOCIATION SOPHIA
'''

# import the necessary packages
import numpy as np
import imutils
import time
import cv2

##########################################################
################## POP-UP VIDEO #########################
#Import the required Libraries
import random
from tkinter import *
from tkinter import ttk
#Create an instance of Tkinter frame
win = Tk()
#Set the geometry of Tkinter frame
def rgb_hack(rgb):
    return "#%02x%02x%02x" % rgb 

win.geometry("1500x270")
win.config(bg=rgb_hack((0, 0, 0))) 
############ define window that opens after clicking ######
def open_popup():
    top= Toplevel(win, bg='#000000')
    top.geometry("1300x500")
    top.title("Résultat")
    result = bool(random.getrandbits(1))
    if result == False:
       text_to_prompt = "Tu n'auras pas ton semestre \n \n Ceci est une simulation. \n \n Dans le monde, des algorithmes prennent \n tous les jours des décisions à nos places. \n Est-ce souhaitable ?"
    else:
        text_to_prompt = "Tu auras ton semestre \n \n Ceci est une simulation. \n \n Dans le monde, des algorithmes prennent \n tous les jours des décisions à nos places. \n Est-ce souhaitable ?"
    Label(top, text= text_to_prompt, bg = '#000000', fg='#00ff00', font=('Mistral 32 bold')).place(x=50,y=80)

##########################################################

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
 
########### CREATE LOGO SPACE IN VIDEO ###########################
logo = cv2.imread('logo_asso')
height = 150
width = 250
logo = cv2.resize(logo, (width, height))
# Create a mask of logo
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
################################################################

################# Start video ###########################
cap = cv2.VideoCapture(0)
time.sleep(2.0)

start = time.time()

# loop over the frames from the video stream
while time.time() < start + 5:
   # grab the frame from the threaded video stream and resize it
   # to have a maximum width of 1500 pixels
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1500)
  
#################################### ADD THE LOGO IN THE VIDEO ######################################
#    Region of Interest (ROI), where we want
#    to insert logo
    roi = frame[50:50 + height, 1200:1200 + width]
  
   # Set an index of where the mask is
    roi[np.where(mask)] = 0
    roi += logo
##########################################################################
# ########### FIND FACE IN THE VIDEO #####################################
   # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
   # pass the blob through the network and obtain the detections and
   # predictions
    net.setInput(blob)
    detections = net.forward()

#####################################################################
##################### Nice effect ##################################
    can = cv2.Canny(frame, (190 / (time.time() - start)) + 1, (200 / (time.time() - start)) + 1)
    rgb_can = cv2.cvtColor(can, cv2.COLOR_GRAY2RGB)
    rgb_can *= np.array((0,1,0),np.uint8)
####################################################################

# #########################################################################
# ############################### SHOW FACE IN THE FRAME ##################
#    # loop over the detections
    for i in range(0, detections.shape[2]):
       
       # extract the confidence (i.e., probability) associated with the
       # prediction
        confidence = detections[0, 0, i, 2]
       # filter out weak detections by ensuring the `confidence` is
       # greater than the minimum confidence
        if confidence < 0.4:
            continue
       # compute the (x, y)-coordinates of the bounding box for the
       # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
       # draw the bounding box of the face along with the associated
       # probability
        text = "{:.2f}%".format(confidence * 100)
 
        y = startY - 10 if startY - 10 > 10 else startY + 10
        y_test = endY + 25 if endY - 25 > 25 else endY + 25
 
        org = (startX, y_test)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 0)
        thickness = 5
 
        cv2.rectangle(frame, (startX, startY), (endX, endY),
           (255, 0, 0), 2)
 
        cv2.putText(frame, "Calcul en cours ... ", org,
           font, fontScale, color, thickness)


    frame[np.where(rgb_can)] = 255
   ##show the output frame
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
   # if the `q` key was pressed, break from the loop
    if key == ord("q"):
       break
# do a bit of cleanup
cv2.destroyAllWindows()
cap.release()


####### Display window that will give the result of the calculation
Label(win, text=" Click sur le bouton ci-dessous pour savoir si tu aura ton semestre", font=('Helvetica 24 bold'), bg = '#000000', fg='#00ff00').pack(pady=20)
Label(win, text= 'Ta performance est calculée \n sur les résultats des étudiant.es précédent.es.', bg = '#000000', fg='#00ff00', font=('Helvetica 24 bold')).place(x=400,y=75)
#Create a button in the main Window to open the popup
ttk.Button(win, text= "Découvrir", command= open_popup).place(x=750,y=175)
win.mainloop()


