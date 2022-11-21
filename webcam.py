######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#
# Author: Evan Juras
# Date: 10/27/19
# Description:
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.

# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util

from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import messaging
import schedule
from multiprocessing import Process

import socket
import pygame
import asyncio

from pycoral.adapters.common import input_size
from pycoral.adapters.detect import get_objects
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from pycoral.utils.edgetpu import run_inference

def msg(aa):
    response = messaging.send(aa)


def speaker():
    while True:
        data = conn.recv(1024)
        getdata = data.decode("utf-8")

        if getdata[2:] == "connect":
            pygame.mixer.init()
            bang = pygame.mixer.Sound("helmet.wav")
            bang.play()

def camera():
    print('Loading {} with {} labels.'.format(args.model, args.labels))
    interpreter = make_interpreter(args.model)
    interpreter.allocate_tensors()
    labels = read_label_file(args.labels)
    inference_size = input_size(interpreter)
    
    countingPrev = 0
    timer = StopWatch()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,448)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    while cap.isOpened():
        t1 = cv2.getTickCount()
        ret, frame = cap.read()
        if not ret:
            print("ok")
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        cv2_im_rgb = cv2.resize(cv2_im_rgb, inference_size)
        run_inference(interpreter, cv2_im_rgb.tobytes())
        objs = get_objects(interpreter, args.threshold)[:args.top_k]
        
        counting = 0
        
        height, width, channels = cv2_im.shape
        scale_x, scale_y = width / inference_size[0], height / inference_size[1]
        
        for obj in objs:
            bbox = obj.bbox.scale(scale_x, scale_y)
            xmin, ymin = int(bbox.xmin), int(bbox.ymin)
            xmax, ymax = int(bbox.xmax), int(bbox.ymax)

            percent = int(100 * obj.score)
            label_name = labels.get(obj.id, obj.id)
            label = '{}% {}'.format(percent, label_name)
            
            if label_name == 'head':
                counting += 1
            
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10)
            
            cv2_im = cv2.rectangle(cv2_im, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.rectangle(cv2_im, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(cv2_im, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
        if counting >= 1:
            if countingPrev < counting:
                cv2.imwrite(filename, frame)

                size = os.path.getsize(filename)
                intSize = size.to_bytes(4, byteorder='big', signed=True)

                conn.send(intSize)

                f = open(filename, 'rb')  # open file as binary
                data = f.read()

                conn.send(data)

                f.flush()
                f.close()

                message = messaging.Message(
                    notification=messaging.Notification(
                        title='안전햇',
                        body='안전모 미착용자' + str(counting) + '명 발생'
                    ),
                    token=registration_token,
                )

                msg(message)
                
                print('안전모 미착용자 ' + str(counting) + '명 발생')
                countingPrev = counting
                timer.start()

            elif countingPrev >= counting:
                time = timer.stop()
                print("메시지 전송 후 " + str(time) + '초 경과')
                if time >= 60:
                    cv2.imwrite(filename, frame)

                    size = os.path.getsize(filename)
                    intSize = size.to_bytes(4, byteorder='big', signed=True)

                    conn.send(intSize)

                    f = open(filename, 'rb')  # open file as binary
                    data = f.read()

                    conn.send(data)

                    f.flush()
                    f.close()

                    message = messaging.Message(
                        notification=messaging.Notification(
                            title='안전햇',
                            body='안전모 미착용자' + str(counting) + '명 발생'
                        ),
                        token=registration_token,
                    )
                    msg(message)
                    
                    print('안전모 미착용자 ' + str(counting) + '명 발생')
                    countingPrev = counting

                    timer.start()

        cv2.moveWindow('frame',150,100)
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0),
                    2, cv2.LINE_AA)
        
        cv2.imshow('frame', cv2_im)
        
        # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        frame_rate_calc = 1 / time1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            conn.close()
            serv.close()
            break

    cap.release()
    cv2.destroyAllWindows()

class StopWatch:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_datetime = None
        self.end_datetime = None

    def start(self):
        self.reset()
        self.start_datetime = datetime.now()

    def stop(self):
        self.end_datetime = datetime.now()
        return self.get_elapsed_seconds()

    def get_elapsed_seconds(self):
        return (self.end_datetime - self.start_datetime).total_seconds()


if __name__ == '__main__':
    HOST = ""
    PORT = 6667
    ADDR = (HOST, PORT)
    BUFSIZE = 4096
    filename = "result/img0.jpg"
    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    serv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # Bind Socket
    serv.bind(ADDR)
    serv.listen(1)
    conn, addr = serv.accept()
    print('client connected ... ', addr)

    cred_path = ""
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
    registration_token = ''

    # Define and parse input arguments
    default_model_dir = 'model'
    default_model = 'edgetpu.tflite'
    default_labels = 'labelmap.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=10,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    
    args = parser.parse_args()

    p1 = Process(target=speaker)
    p2 = Process(target=camera)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
