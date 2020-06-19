import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os
from mdl import Net
    
def load_model(path):
    model=Net()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model



def load_transforms(image):
    data=[]
    image=image.resize((30,30))
    image=np.array(image)
    data.append(image)
    imgs=torch.from_numpy(np.transpose(data,(0,3,1,2))).float()
    return imgs

def predict(net,transform,image):
    classes=["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)",
         "Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vechiles over 3.5 metric tons","Right-of-way at the next intersection","Priority road",
         "Yield","Stop","No vechiles","Vechiles over 3.5 metric tons prohibited",
         "No entry","General caution","Dangerous curve to the left","Dangerous curve to the right",
         "Double curve","Bumpy road","Slippery road","Road narrows on the right",
         "Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing",
         "Beware of ice/snow","Wild animals crossing","End of all speed and passing limits",
         "Turn right ahead","Turn left ahead","Ahead only","Go straight or right",
         "Go straight or left","Keep right","Keep left","Roundabout mandatory",
         "End of no passing","End of no passing by vechiles over 3.5 metric tons"]
    input_img=load_transforms(image)
    output=net(input_img)
    prob,pred=torch.max(output,1)
    pred=pred.numpy()
    return classes[pred[0]]
    
