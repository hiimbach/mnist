import os 
import sys
ROOT = os.getcwd()
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch 
import torchvision.transforms as tf
import cv2

from model.model import ConvNet 
from utils.cv2_to_np import cv2_to_array


class Classifier():
    def __init__(self, weight_path = 'weight/weight.pt') -> None:
        self.net = ConvNet()
        self.net.load_state_dict(torch.load(weight_path))

        self.transform = tf.Compose([tf.ToTensor(),
                                tf.Normalize((0.1307,), (0.3081,))
        ])

    def predict(self, img_path):
        img = cv2.imread(img_path, 0)

        # Convert img to np array
        arr = cv2_to_array(img)
        
        print(arr.shape)
        
        with torch.no_grad():
            out = self.net(self.transform(arr))
            pred = int(torch.max(out, dim=1)[1][0])
        
        return pred
    

if __name__ == "__main__":
    classifier = Classifier()
    classifier.predict('data/5.1.png')
    
    import ipdb; ipdb.set_trace()