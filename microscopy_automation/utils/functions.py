import cv2
import numpy as np

def getBoundingBox(img, mask, prc = 0.2):
    _, contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    coor = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x = x-int(w*prc) if x-int(w*prc) >= int(mask.shape[1]*0.001) else int(mask.shape[1]*0.001)
        y = y-int(h*prc) if y-int(h*prc) >= int(mask.shape[0]*0.001) else int(mask.shape[0]*0.001)
        w = w+2*int(w*prc) if x + w+2*int(w*prc) <= int(mask.shape[1]*0.999) else int(mask.shape[1]*0.999) - x
        h = h+2*int(h*prc) if y + h+2*int(h*prc) <= int(mask.shape[0]*0.999) else int(mask.shape[0]*0.999) - y
        coor.append((y, x, h, w))
    
    return coor

def cutLeukocyte(img, coor_bnb):
    cutted = []
    for (y,x,h,w) in coor_bnb:
        cutted.append(img[y:y+h, x:x+w])
    
    return np.array(cutted)

def colorBnbCls(img, coor_bnb, classes):
    for coor, label in zip(coor_bnb, classes):
        img = cv2.rectangle(img, (coor[1], coor[0]), (coor[1] + coor[3], coor[0] + coor[2]), (255,0,0), 4)
        img = cv2.putText(img, "{}, proba: {}".format(label['class'], round(label['proba'], 3)), (coor[1]+2, coor[0]+coor[2]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2) 