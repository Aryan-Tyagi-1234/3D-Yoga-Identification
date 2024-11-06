import cv2
import mediapipe as mp
import numpy as np
import h5py
import os

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

datalist={0: 'adho mukha svanasana', 1: 'adho mukha vriksasana', 2: 'akarna dhanurasana', 3: 'anantasana', 4: 'anjaneyasana', 5: 'ardha chandrasana', 6: 'ashtanga namaskara', 7: 'baddha konasana', 8: 'bakasana', 9: 'bhujangasana', 10: 'bitilasana', 11: 'chaturanga dandasana', 12: 'dandasana', 13: 'dhanurasana', 14: 'dwi pada viparita dandasana', 15: 'halasana', 16: 'hanumanasana', 17: 'karnapidasana', 18: 'natarajasana', 19: 'parighasana', 20: 'parivrtta trikonasana', 21: 'pincha mayurasana', 22: 'ustrasana', 23: 'utkatasana', 24: 'utthita hasta padangustasana', 25: 'virabhadrasana ii', 26: 'virabhadrasana iii', 27: 'vriksasana'}

arr1=[]
arr2=[]
arr3=[]
arr4=[]

for i in range(28):
    with mp_pose.Pose(static_image_mode=True,model_complexity=2,enable_segmentation=True,min_detection_confidence=0.5) as pose:
        folder=os.listdir("MAIN1/{}".format(datalist[i]))
        
        j=1
        for photo in folder:
            image = cv2.imread("MAIN1/{}/{}".format(datalist[i],photo))
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            lt=checkAccuracy(results)
            if lt:
                if j%21!=0:
                    arr1+=lt
                    arr2+=[i]
                else:
                    arr3+=lt
                    arr4+=[i]
                j+=1
        print(datalist[i],"done")
arr2=np.array(arr2)
arr4=np.array(arr4)
arr1=np.array(arr1).reshape(arr2.shape[0],72)
arr3=np.array(arr3).reshape((arr4.shape[0],72))
    


print(type(arr1))
print(type(arr2))
print(arr1.shape)
print(arr2.shape)              
print(type(arr3))
print(type(arr4))
print(arr3.shape)
print(arr4.shape) 
              

with h5py.File('yogaDataset3D.h5', 'w') as f:
    f.create_dataset('X_train', data = arr1)
    f.create_dataset('y_train', data = arr2)        
    f.create_dataset('X_test', data = arr3)
    f.create_dataset('y_test', data = arr4)     
cv2.destroyAllWindows()
print("done")
