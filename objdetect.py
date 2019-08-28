import cv2
import time
import numpy as np
from multiprocessing import Process
from multiprocessing import Queue
import argparse
#from os.path import exists, isfile

import edgetpu.detection.engine
#from edgetpu.utils import image_processing
from PIL import Image

#cv2 capture (video or camera) 
class MyCap:
    def __init__(self,filename, resolution=None):
        print('Trying to open Videofile ' + filename)
        self.cap = cv2.VideoCapture(filename)
        print("[video info] W, H, FPS, frames")
        print(self.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(self.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(self.get(cv2.CAP_PROP_FPS))
        print(self.get(cv2.CAP_PROP_FRAME_COUNT))
        if resolution is not None:
            res = resolution.split('x')
            self.setResolution(int(res[0]),int(res[1]))      
        else:
            self.setResolution(int(self.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.get(cv2.CAP_PROP_FRAME_HEIGHT)))    
        print ("[Displaying Resolution] " + str(self.frameWidth) + "x" + str(self.frameHeight))

    def setResolution(self,width,heigth):
        self.frameWidth = width
        self.frameHeight = heigth

    def get(self,attribute):
        return self.cap.get(attribute)
        
    def set(self,attribute,value):
        return self.cap.set(attribute,value)
        
    def isOpened(self):
        return self.cap.isOpened()
        
    def read(self):
        ret, frame =  self.cap.read()
        if ret == False:
            return None
        return cv2.resize(frame, (self.frameWidth,self.frameHeight))
        #Image.fromarray(resize)
        
        


#use multiprocessing to feed data into Coral USB stick and wating for result
class CoralDetector:
    inputQueue = None
    outputQueue = None
    engine = None
    labels = None
    
    def __init__(self):
        # initialize queues: 
        # inputqueue for captured frames
        # outputqueue for detected results
        # both of size 1 since this is soft real time (always work on the most current frame)
        self.inputQueue = Queue()
        self.outputQueue = Queue()

        #initialize TPU engine 
        self.engine = edgetpu.detection.engine.DetectionEngine('tpu_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
        self.initlabels()
        
        print("[INFO] starting classification process...")
        p = Process(target=self.classify_frame)
        p.daemon = True
        p.start()        

    def initlabels(self):
        labels_file = 'tpu_models/coco_labels.txt'
        # Read labels from text files. Note, there are missing items from the coco_labels txt hence this!
        self.labels = [None] * 10
        with open(labels_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split(maxsplit=1)
            self.labels.insert(int(parts[0]),str(parts[1])) 
            
        print("[COCO LABELS] " + str(self.labels))
    

    #the actual inference process
    def classify_frame(self):
        while True:
            # grab the latest frame from the input queue
            img = self.inputQueue.get() #blocking on empty queue
            while not self.inputQueue.empty():
                img = self.inputQueue.get()
            if img is "EXIT":
                return
                    
            #shove it to CORAL USB stick and wait for the results 
            results = self.engine.DetectWithImage(img, threshold=0.4, keep_aspect_ratio=True, relative_coord=False, top_k=10)

            data_out = []

            if results:
                for obj in results:
                    inference = []
                    box = obj.bounding_box.flatten().tolist()
                    xmin = int(box[0])
                    ymin = int(box[1])
                    xmax = int(box[2])
                    ymax = int(box[3])

                    inference.extend((self.labels[obj.label_id],obj.score,xmin,ymin,xmax,ymax))
                    data_out.append(inference)
            self.outputQueue.put(data_out)

    #returns inference objects from last detection run or None
    def get(self):
        out = None
        while not self.outputQueue.empty():
            out = self.outputQueue.get()
        return out
    
    #puts img into inference queue
    def put(self,img):
        self.inputQueue.put(img)

                                
    def terminate(self):
        self.inputQueue.put("EXIT")                            


#display the resulting image with the detections
class Frame:
    def __init__(self):
        self.confThreshold = 0.6
    
    def run(self, cap, detector):
        #time the frame rate....
        timer1 = time.time()
        timer2 = 0
        frames = 0
        inferences = 0
        fps = 0.0
        qfps = 0.0
        while(cap.isOpened()):
            # Capture frame-by-frame
            frame = cap.read()
                
            if frame:
                if inferences == 1:
                    timer2 = time.time()
        
                #calssify current frame (non-blocking)
                detector.put(Image.fromarray(frame))
                #grab the detections (non-blocking)
                out = detector.get()
        
                if out is not None:
                    # loop over the detections
                    for detection in out:
                        labeltxt = detection[0]
                        confidence = detection[1]
                        xmin = detection[2]
                        ymin = detection[3]
                        xmax = detection[4]
                        ymax = detection[5]
                        if confidence > self.confThreshold:
                            #bounding box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))
                            #label
                            labLen = len(labeltxt)*5+40
                            cv2.rectangle(frame, (xmin-1, ymin-1), (xmin+labLen, ymin-10), (0,255,255), -1)
                            #labeltext
                            cv2.putText(frame,' '+labeltxt+' '+str(round(confidence,2)), (xmin,ymin-2), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,0),1,cv2.LINE_AA)
                
                    inferences += 1
        
                vidpos = round(cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100)
                # Display the resulting frame
                cv2.rectangle(frame, (0,0), (cap.frameWidth,20), (0,0,0), -1)
        
                cv2.rectangle(frame, (0,cap.frameHeight-20), (cap.frameWidth,cap.frameHeight), (0,0,0), -1)
                cv2.putText(frame,'Threshold: '+str(round(self.confThreshold,1)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
        
                cv2.putText(frame,'VID FPS: '+str(fps), (cap.frameWidth-80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
        
                cv2.putText(frame,'TPU FPS: '+str(qfps), (cap.frameWidth-80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
        
                cv2.putText(frame,'Video pos: '+str(vidpos), (10, cap.frameHeight-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
                
        
                cv2.namedWindow('Coral',cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Coral',cap.frameWidth,cap.frameHeight)
                cv2.imshow('Coral',frame)
                
                # FPS calculation
                frames += 1
                if frames >= 1:
                    end1 = time.time()
                    t1secs = end1-timer1
                    fps = round(frames/t1secs,2)
                if inferences > 1:
                    end2 = time.time()
                    t2secs = end2-timer2
                    qfps = round(inferences/t2secs,2)
        
        
                keyPress = cv2.waitKey(1) #Altering waitkey val can alter the framreate for vid files.
                if keyPress == ord('q'):
                    return
                if keyPress == 82 and self.confThreshold < 1:
                    self.confThreshold += 0.1
                if keyPress == 84 and self.confThreshold > 0.4:
                    self.confThreshold -= 0.1
                if keyPress == 83:
                    #print ("Curr pos: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    framelapse = cap.get(cv2.CAP_PROP_FPS) *10 #10 seconds
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) + framelapse < cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + framelapse)
                    else:
                        return
                if keyPress == 81:
                    #print ("Curr pos: " + str(cap.get(cv2.CAP_PROP_POS_FRAMES)))
                    framelapse = cap.get(cv2.CAP_PROP_FPS) *10 #10 seconds
                    if cap.get(cv2.CAP_PROP_POS_FRAMES) - framelapse > 0:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - framelapse)
                    else:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)    
                #print (str(keyPress))
            else: 
                return        


if __name__== "__main__":
    #parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument("-r", "--resolution", help="Video resolution widthxheigth (e.g. 640x480)")
    args = parser.parse_args()
    
        
    #init video stream, detector and frame display
    cap = MyCap(args.filename, args.resolution)
    detector = CoralDetector()
    frame = Frame()
    frame.run(cap, detector)
    
    detector.terminate()
    cap.cap.release()
    cv2.destroyAllWindows()


