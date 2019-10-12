from torchvision import transforms
import cv2
from PIL import Image
import numpy as np

class Resize:
    """Resize image to PIC_SIZE
    
    Args:
        PIC_SIZE ((X_SIZE, Y_SIZE)): shape image in exit
    
    """
    
    def __init__(self, PIC_SIZE):
        self.PIC_SIZE = PIC_SIZE
    
    def __call__(self, sample):
        if isinstance(sample, np.ndarray):
            return cv2.resize(sample, self.PIC_SIZE)
        if isinstance(sample, Image.Image):
            return sample.resize((self.PIC_SIZE, self.PIC_SIZE))

class ToPILImage:

    def __init__(self):
        pass
    
    def __call__(self, sample):
        return Image.fromarray(sample)

    

def get_preprocessing_for_segmentation(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: transforms.Compose
    
    """
    
    _transform = transforms.Compose([Resize((256,256)), 
                                     preprocessing_fn,
                                     transforms.ToTensor()])
    return _transform

def get_preprocessing_for_classification():
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: transforms.Compose
    
    """
    
    _transform = transforms.Compose([
                                    # ToPILImage,
                                    Resize((256,256)), 
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return _transform

def preprocessing_for_predict_mask(pr_mask, pic_size, threshold = 0.5, iterations_for_dilate=2):
    #resize
    pr_mask = cv2.resize(pr_mask, (pic_size[1], pic_size[0]))

    #threshold
    pr_mask[pr_mask >= threshold] = 255
    pr_mask[pr_mask < threshold] = 0
    
    #dilate
    kernel = np.ones((5,5),np.uint8)
    pr_mask = pr_mask.astype('uint8')
    pr_mask = cv2.morphologyEx(pr_mask, cv2.MORPH_OPEN, kernel, iterations = 3)
    pr_mask = cv2.dilate(pr_mask,kernel, iterations = iterations_for_dilate)
    return pr_mask

