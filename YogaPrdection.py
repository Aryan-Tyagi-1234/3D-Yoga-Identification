import cv2
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model
import mediapipe as mp
import math as m
import numpy as np
import tensorflow as tf 
if tf.test.gpu_device_name(): 

    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
   
def checkAccuracy(results):
      
   if results.pose_landmarks:
       # right leg
       rl1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].z]
       rl2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].z]
       rl3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].z]
       
       # left leg
       ll1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].z]
       ll2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].z]
       ll3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].z]
       
       # right arm
       ra1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].z]
       ra2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].z]
       ra3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z]
       
       # left arm
       la1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].z]
       la2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].z]
       la3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z]
             
       # ear
       le=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z]
       re=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z]
       
       # right hand
       rh1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].z]
       rh2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].z]
       rh3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].z]
       
       # LEFt hand
       lh1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].z]
       lh2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].z]
       lh3=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].z]
       
       # left feet
       lf1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].z]
       lf2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].z]
       
       # right feet
       rf1=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].z]
       rf2=[results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].z]
       
       lt=[le,re,la1,ra1,la2,ra2,la3,ra3,lh3,rh3,lh2,rh2,lh1,rh1,ll1,rl1,ll2,rl2,ll3,rl3,lf2,rf2,lf1,rf1]
       lt1=[]
       
       x=0
       for i in range(24):
           if x>lt[i][0]:
               x=lt[i][0]
       for i in range(24):
           lt[i][0]-=x        
       
       y=0
       for i in range(24):
           if y>lt[i][1]:
               y=lt[i][1]
       for i in range(24):
           lt[i][1]-=y
            
       z=0
       for i in range(24):
           if z>lt[i][2]:
               z=lt[i][2]
       for i in range(24):
           lt[i][2]-=z
       
       for i in lt:
           lt1+=i

       return lt1


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

vid = cv2.VideoCapture("bitilasana.mp4")
with mp_pose.Pose(min_detection_confidence=0.5,min_tracking_confidence=0.5) as pose:
  ret, image = vid.read()
  while ret :
    ret, image = vid.read()
    class_label="No Yoga"
    if ret==False:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      break
    
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    #image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    lt=checkAccuracy(results)
    if lt:
        lt=np.array(lt)
        lt=np.reshape(lt,(1,6, 4, 3, 1))
    
        m2=load_model("YogaPrediction3D.h5")    ##### 29
        pred2=m2(lt)
        pred2=np.array(pred2)
        pred2=pred2.argmax()
        yoga2={0: 'Z', 1: 'adho mukha svanasana', 2: 'adho mukha vriksasana', 3: 'akarna dhanurasana', 4: 'anantasana', 5: 'anjaneyasana', 6: 'ardha chandrasana', 7: 'ashtanga namaskara', 8: 'baddha konasana', 9: 'bakasana', 10: 'bhujangasana', 11: 'bitilasana', 12: 'chaturanga dandasana', 13: 'dandasana', 14: 'dhanurasana', 15: 'dwi pada viparita dandasana', 16: 'halasana', 17: 'hanumanasana', 18: 'karnapidasana', 19: 'natarajasana', 20: 'parighasana', 21: 'parivrtta trikonasana', 22: 'pincha mayurasana', 23: 'ustrasana', 24: 'utkatasana', 25: 'utthita hasta padangustasana', 26: 'virabhadrasana ii', 27: 'virabhadrasana iii', 28: 'vriksasana'}
        class_label=yoga2[pred2+1]
    
    x=list(image.shape)
    font=cv2.FONT_HERSHEY_COMPLEX
    org=(int(x[1]/2),40)
    font_size=1
    thick=2
    color=(255,0,0) 

    cv2.putText(image,class_label, org, font, 1, color,thick,cv2.LINE_AA)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    #image=cv2.flip(image, 1)
    cv2.imshow('MediaPipe Pose', image)
   
    if cv2.waitKey(1) == ord(' '):
        print("done")
        break
      
vid.release()
cv2.destroyAllWindows() 
print("done")