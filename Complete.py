import csv
import pandas as pd
import ast
import numpy as np
from os import listdir
from os.path import isfile, join,split
import speech_recognition as sr
import pyttsx3
import cv2
import pytesseract
from skimage.filters import threshold_local
from PIL import Image
from googletrans import Translator
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
#path of haarcascade_frontal_face_default.xml
face_classifier=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

def Save_new_face():
    name='unknown'
    while(1):
        SpeakText('What is the Name?')
        name=AudioConversion()
        SpeakText('Is it '+name)
        result=AudioConversion()
        if(result.__contains__('yes')):
            break;
    def face_extract(img):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=face_classifier.detectMultiScale(gray,1.3,5)
        if faces == ():
            return None
        for (x,y,w,h) in faces:
            cropped_face = img[y:y+h , x:x+w]
        return cropped_face
    cap=cv2.VideoCapture(0)
    count=0
    dict = {}
    mx = int(0)
    keys = pd.read_csv('data/dictionary_string.csv', header=None, index_col=0, squeeze=True).to_dict()
    for row in keys:
        dict[int(row)] = keys[row]
        if (mx < int(row)):
            mx = int(row)
    mx = mx + 1
    dict[mx] = name
    w = csv.writer(open("data/dictionary_string.csv", "w"))
    for key, val in dict.items():
        w.writerow([key, val])
    while True:
        time.sleep(3)
        ret,frame=cap.read()
        count += 1
        if face_extract(frame) is not None:
            face=cv2.resize(face_extract(frame),(200,200))
            face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            #path where you want to keep the Faces dataset
            path='data/user/.' +str(mx)+'.'+ str(count) + '.jpg'
            cv2.imwrite(path,face)
            cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.imshow('Face',face)
        else:
            print('No face')
            pass
        SpeakText(str(count)+ ' Face added , Press Button to Exit')
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27 or count==15:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('samples collected')
    SpeakText('samples collected. Training the data')
    Train_faces()

def Train_faces():
    #path of Dataset
    data_path = 'data/user/'
    files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    trains, label_s = [], []
    for i, file in enumerate(files):
        im_path = data_path + files[i]
        img = cv2.imread(im_path, cv2.IMREAD_GRAYSCALE)
        trains.append(np.asarray(img, dtype=np.uint8))
        id=int(split(im_path)[-1].split(".")[1])
        label_s.append(id)
    label_s = np.asarray(label_s, dtype=np.int)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(trains), np.asarray(label_s))
    #path where you want to keep the trained dataset
    model.write('data/trainer/trainer.yml')
    SpeakText('Model Trained. ')
    recog_face()

def recog_face():
    dict = {}
    #dictionary contains details of Faces saved
    keys = pd.read_csv('data/dictionary_string.csv', header=None, index_col=0, squeeze=True).to_dict()
    count=0
    for row in keys:
        dict[int(row)] = keys[row]
        count+=1
    if(count==1):
        SpeakText('Empty Dataset')
        return
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    #path of trained dataset
    recognizer.read('data/trainer/trainer.yml')
    #path of Haarcascade
    faceCascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml');
    font = cv2.FONT_HERSHEY_SIMPLEX
    id = 0
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height
    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)
    oldname=""
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray,1.2,5)
        if faces is None:
            oldname="No Face"
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            if (confidence < 60 and confidence > 0):
                id = dict[id]
                confidence = "  {0}%".format(round(100 - confidence))
                if(oldname!=str(id)):
                    SpeakText(str(id)+' is in front of you')
                    oldname=str(id)
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)
        cv2.imshow('camera', img)
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    cam.release()
    cv2.destroyAllWindows()

def newyolo():
    def get_output_layers(net):
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])
        color = COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        SpeakText(label)
    cap=cv2.VideoCapture(0)
    while(1):
        ret,image=cap.read()
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392
        classes = None
        with open('data/yolov3.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
        # can be downloaded from here https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
        net = cv2.dnn.readNet('data/yolov3.weights', 'data/yolov3.cfg')
        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        cv2.imshow("object detection", image)
        SpeakText("Identified")
        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
    cv2.imwrite("object-detection.jpg", image)
    cv2.destroyAllWindows()

def bookread():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(3)
    #path where tesseract is saved
    pytesseract.pytesseract.tesseract_cmd = 'data/tessereact/tesseract.exe'
    while (1):
        SpeakText('Bring the book close to get the image:')
        time.sleep(3)
        ret, image = cap.read()
        #gray=image
        #cv2.bitwise_not(image, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #gray= cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
        #image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        #text = pytesseract.image_to_string(gray)
        #print("without ngtive is:  "+text)
        text = pytesseract.image_to_string(image)
        #print(""+text)
        SpeakText(text)
        cv2.imshow('image',image)
        #cv2.imshow('gray',gray)
        SpeakText("If u do not want to continue press button")
        if(cv2.waitKey(1)==27):
            break
    cap.release()
    cv2.destroyAllWindows()

def SpeakText(command):
    print(command)
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

def AudioConversion():
    r = sr.Recognizer()
    while (1):
        try:
            with sr.Microphone() as source2:
                r.adjust_for_ambient_noise(source2, duration=0.1)
                print('Listening.....')
                audio2 = r.listen(source2)
                MyText = r.recognize_google(audio2)
                #print("Did you say " + MyText)
                return MyText
        except sr.RequestError as e:
            #print("Could not request results; {0}".format(e))
            SpeakText('Sorry I did not get that')
            return AudioConversion()
        except sr.UnknownValueError:
            #print("unknown error occured")
            SpeakText('Sorry I did not get that')
            return AudioConversion()

SpeakText('Device is On now')

while(1):
    SpeakText('Which operation to run?.....Save new face.....Face Recognition.....Object Detection......Book Reading......')
    operation=AudioConversion()
    SpeakText('Did u say'+operation)
    result=AudioConversion()
    if(result.__contains__('yes') and operation.__contains__('save')):
        Save_new_face()
    elif(result.__contains__('yes') and operation.__contains__('face')):
        recog_face()
    elif(result.__contains__('yes') and operation.__contains__('object')):
        newyolo()
    elif(result.__contains__('yes') and operation.__contains__('book')):
        bookread()
    else:
        SpeakText('Sorry Could not get that')
    SpeakText('Do u want to continue:')
    resultt=AudioConversion()
    if(resultt.__contains__('no') or resultt==''):
        SpeakText('Adios Amigo . Powering Off!!')
        break
