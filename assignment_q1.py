import numpy as np
import cv2
import pathlib
from scipy import misc
import os





count = 0
for k in range(12):
    count=0
    for n in range(3):
        for i in range(4,24):
            for j in range(4,24):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-3,j-1),(i+3,j-1),(0,0,255),1)
                a=cv2.line(a,(i-3,j),(i+3,j),(0,0,255),1)
                a=cv2.line(a,(i-3,j+1),(i+3,j+1),(0,0,255),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_3_%i_R'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_3_%i_R'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'7_3_%i_R_%i.jpg'%((15*k),count)),a2)
                count = count+1

count = 0
for k in range(12):
    count=0
    for n in range(3):
        for i in range(4,24):
            for j in range(4,24):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-3,j-1),(i+3,j-1),(255,0,0),1)
                a=cv2.line(a,(i-3,j),(i+3,j),(255,0,0),1)
                a=cv2.line(a,(i-3,j+1),(i+3,j+1),(255,0,0),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_3_%i_B'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_3_%i_B'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'7_3_%i_B_%i.jpg'%((15*k),count)),a2)
                count = count+1

count = 0
for k in range(12):
    count=0
    for n in range(7):
        for i in range(8,20):
            for j in range(8,20):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-7,j-1),(i+7,j-1),(255,0,0),1)
                a=cv2.line(a,(i-7,j),(i+7,j),(255,0,0),1)
                a=cv2.line(a,(i-7,j+1),(i+7,j+1),(255,0,0),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_3_%i_B'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_3_%i_B'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'15_3_%i_B_%i.jpg'%((15*k),count)),a2)
                count = count+1


count = 0
for k in range(12):
    count=0
    for n in range(7):
        for i in range(8,20):
            for j in range(8,20):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-7,j-1),(i+7,j-1),(0,0,255),1)
                a=cv2.line(a,(i-7,j),(i+7,j),(0,0,255),1)
                a=cv2.line(a,(i-7,j+1),(i+7,j+1),(0,0,255),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_3_%i_R'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_3_%i_R'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'15_3_%i_R_%i.jpg'%((15*k),count)),a2)
                count = count+1

count = 0
for k in range(12):
    count=0
    for n in range(7):
        for i in range(8,20):
            for j in range(8,20):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-7,j),(i+7,j),(0,0,255),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_1_%i_R'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_1_%i_R'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'15_1_%i_R_%i.jpg'%((15*k),count)),a2)
                count = count+1


count = 0
for k in range(12):
    count=0
    for n in range(7):
        for i in range(8,20):
            for j in range(8,20):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-7,j),(i+7,j),(255,0,0),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_1_%i_B'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/15_1_%i_B'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'15_1_%i_B_%i.jpg'%((15*k),count)),a2)
                count = count+1


count = 0
for k in range(12):
    count=0
    for n in range(3):
        for i in range(4,24):
            for j in range(4,24):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-3,j),(i+3,j),(0,0,255),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_1_%i_R'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_1_%i_R'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'7_1_%i_R_%i.jpg'%((15*k),count)),a2)
                count = count+1


count = 0
for k in range(12):
    count=0
    for n in range(3):
        for i in range(4,24):
            for j in range(4,24):
                #if (k==1):
                zero_mat=np.zeros((28,28,3))
                a=cv2.line(zero_mat,(i-3,j),(i+3,j),(255,0,0),1)
                a2 = misc.imrotate(a, 15*k, 'nearest')
                pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_1_%i_B'%(15*k)).mkdir(parents=True, exist_ok=True)
                p = pathlib.Path('/home/souvik/Desktop/DEEP_learning/assignment_1/7_1_%i_B'%(15*k)).stem
                cv2.imwrite(os.path.join(p,'7_1_%i_B_%i.jpg'%((15*k),count)),a2)
                count = count+1