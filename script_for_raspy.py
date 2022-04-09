#importamos las librerias para la dereccion
#y tambien las necesarias para tflite

import numpy as np
#import pygame
from subprocess import Popen
#import tensorflow as tf
import cv2
import os
import importlib.util
from threading import Thread
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from threading import Thread
import time

#! pip install playsound
#! pip install pygobject #esto no se porque
#from playsound import playsound #da problemas en la raspberry


# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True
        #self.stream.release()


#por último hacemos un bucle constante que prediga sobre la imagen y la enseñe

#vamos a redefinir todo lo del modelo para que esta celda sea autocontenida
#directorio de los modelos y los datos
dir_model="./modelos"
dir_data="./datos"

labels="coco_labels.txt"
model_name_tpu="tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite"
model_name_def="tf2_ssd_mobilenet_v2_coco17_ptq.tflite"


#vamos a crear el interprete para correrlo con la tpu o sin ella
use_TPU=True
from tflite_runtime.interpreter import Interpreter
if use_TPU:
    from tflite_runtime.interpreter import load_delegate
# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    GRAPH_NAME = 'edgetpu.tflite' 
#ahora vamos a cargar el modelo

if use_TPU:
    interpreter=Interpreter(model_path=os.path.join(dir_model,model_name_tpu),
                           experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    #ambos funcionan, no se muy bien la diferencia del que viene de tflite_runtime del otro
    #pero si es importante que el modelo aqui sea el no edge.tpu
    interpreter=Interpreter(model_path=os.path.join(dir_model,model_name_def))
    #interpreter = tf.lite.Interpreter(model_path=os.path.join(dir_model,model_name))
    
#añadimos esta linea como para "inicializar con tensores el modelo"
interpreter.allocate_tensors()
    
#y de este modelo que acabamos de cargar tenemos que averiguar que parámetros nos requiere
input_details = interpreter.get_input_details()
#y que parametros nos devuelve. Esto lo usaremos mas adelante, tras declarar y ejecutar el modelo
output_details = interpreter.get_output_details()


#y las labels

with open('./modelos/coco_labels.txt') as f:
    coco_classes = np.array(f.read().splitlines())
    
width, height=(input_details[0]["shape"][1],input_details[0]["shape"][2])


resW,resH=640,480
videostream = VideoStream(resolution=(resW,resH),framerate=30).start()
time.sleep(1)

#cargamos el sonido
#pygame.mixer.init()
#pygame.mixer.music.load("./datos/nogod_crop.mp3")


#añadimos unas lineas para calcular el framerate
min_conf_threshold=0.7

imW, imH = int(resW), int(resH)
frame_rate_calc = 1
freq = cv2.getTickFrequency()

#con un buen framerate, vamos a registrar todas las lecturas cada 300 detecciones, si phone esta mas de la mitad,
#ejecutamos un audio
last_detections=[]
palabra="cell phone"
umbral1=20
umbral2=umbral1//2

#añadimos un contador para que deje unos pocos hasta la proxima vez
contar_repe=0 #esto queda pendiente
while True:

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame1 = videostream.read()
    frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

    boxes = interpreter.get_tensor(output_details[1]['index'])[0]

    num = interpreter.get_tensor(output_details[2]['index'])[0]

    classes = interpreter.get_tensor(output_details[3]['index'])[0]

    
    # Loop over all detections and draw detection box if confidence is above minimum threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # Get bounding box coordinates and draw box
            # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            #aqui sacamos el area relativa del rectangulo
            rel_area=(boxes[i][2]-boxes[i][0])*(boxes[i][3]-boxes[i][1])
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            # Draw label
            object_name = coco_classes[int(classes[i])] # Look up object name from "labels" array using class index
            #guardamos las ultimas detecciones
            last_detections.append(str(object_name))
            
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
    #cada 300 frames, comprobamos si una palabra clave esta almenos la mitad de las veces en la lista de
    #objetos detectados, y cada 300 volvemos a empezar
    if len(last_detections)>umbral1:
        last_detections=[]
    elif len(last_detections)!=0:
        veces_repe=sum([palabra==i for i in last_detections])
        if veces_repe>umbral2:
            #playsound("./datos/nogod_crop.mp3",block=False)
            #pygame.mixer.music.play()
            #while pygame.mixer.music.get_busy() == True:
            #   continue
            
            #aqui ponemos la comprobación del area
            if rel_area<(0.4*0.4):
                player = Popen(["mplayer", "./datos/oh_no2_crop.mp3"])
                print("oh no")		
            else:
                player = Popen(["mplayer", "./datos/nogod_crop.mp3"])
                print("NO GOD NO")
            last_detections=[]
            #player.stdin.write("q")



    
  
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    umbral1=frame_rate_calc #esto lo metemos para que tome de umbral de detección los frames
    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()
