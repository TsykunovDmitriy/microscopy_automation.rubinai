import torch
import cv2
from PIL import Image
from torchvision import models
import segmentation_models_pytorch as smp

class SegmentationModel:

    def __init__(self, path, device='cpu'):
        
        self.path = path
        self.encoder_name = 'resnet34'
        self.device = device
        
        self.model = torch.load(self.path, map_location=self.device)
        self.model.eval()

    def _to_device(self, device):
        self.model.to(device)
        self.device = device

    def predict(self, inputs):
        inputs = inputs.float().to(self.device).unsqueeze(0)
        predict_tensor = self.model(inputs)
        predict = predict_tensor.squeeze().cpu().detach().numpy()
        return predict
    
    def get_preprocessing_fn(self):
        return smp.encoders.get_preprocessing_fn(self.encoder_name, 'imagenet')

class ClassificationModel:

    def __init__(self, path, classes=['BASOPHILE', 'EOSINOPHILE', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL'], device='cpu'):
        
        self.path = path
        self.device = device
        self.classes = classes

        self.model = torch.load(self.path, map_location = self.device)
        self.model.eval()
    
    def _to_device(self, device):
        self.model.to(device)
        self.device = device

    def predict(self, inputs, preprocessing_fn):
        inputs = preprocessing_fn(inputs)
        inputs = inputs.unsqueeze(0)
        output = self.model(inputs)
        _, predicted = torch.max(output, 1)
        proba = torch.nn.Softmax(dim=1)(output)
        predicted = predicted.cpu().data.numpy()[0]
        proba = max(proba.cpu().data.numpy()[0])
        return {'class': self.classes[predicted], 'proba': float(proba)}



    


    

    

    
