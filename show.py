import cv2
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf

from model.model import ConvNet


def most_frequent(l):
    max_ele = 0
    char = 0
    for i in set(l):
        if max_ele < l.count(i):
            char = i
            max_ele = l.count(i)
    
    return char
    
net = ConvNet()
net.load_state_dict(torch.load('weight.pt'))
transform = tf.Compose([tf.ToTensor(),
                        tf.Normalize((0.1307,), (0.3081,))
])

vid = cv2.VideoCapture(0)
history = [0]*12

while True:
    ret, frame = vid.read()
    
    inp = cv2.cvtColor(frame[150:250, 420:520], cv2.COLOR_BGR2GRAY)
    
    # Preprocess
    inp = cv2.resize(inp, dsize=[28,28])
    inp = 255-inp
    inp = cv2.threshold(inp, 170, 255, cv2.THRESH_TOZERO)[1]
    arr = np.asarray(inp)
    
    cv2.imshow('crop', inp)
    
    with torch.no_grad():
        out = net(transform(arr))
        # print(torch.max(out, dim=1))
        pred = int(torch.max(out, dim=1)[1][0])
        
    history.pop(0)
    history.append(pred)
    print(history)
    
    cv2.putText(frame, str(most_frequent(history)), (400, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)        
    cv2.rectangle(frame, (420, 150), (520, 250), (0,255,0), 2)
    cv2.imshow('frame', frame)
    
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
    