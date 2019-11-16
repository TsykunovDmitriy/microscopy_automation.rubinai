# Leukocyte microscopy automation project
Package for automating leukocyte microscopy implemented in Python with used PyTorch deep learning framework.

<p align="center">
  <img width="400px" src="https://raw.githubusercontent.com/TsykunovDmitriy/microscopy_automation.rubinai/master/rubinai.jpg">
  <img width="400px" src="https://raw.githubusercontent.com/TsykunovDmitriy/microscopy_automation.rubinai/master/bmstu.png">
</p>

This application was created as part of a decision support system for the microscopy automation project RubinAI. RubinAI is project carried out at the department of biomedical technical systems at Bauman Moscow State Technical University.

At the heart of the working there are two convolutional neural networks. First one is for segmentation, second one is for classification.
The pipeline of detection and classification of leukocytes is presented in the figure below. 

![alt text](https://raw.githubusercontent.com/TsykunovDmitriy/microscopy_automation.rubinai/master/pipeline_demonstration_1.png)
![alt text](https://raw.githubusercontent.com/TsykunovDmitriy/microscopy_automation.rubinai/master/pipeline_demonstration_2.png)

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Install needed package
Used version of python - 3.7.2 (we recommend using virtualenv). 

First need to install some libraries for image processing and deep learning framework. All packages that you need to install described in requirements.txt.

```
pip3 install -r requirements.txt
```

### Starting demo
All files that you want to process must be in the same folder (ex. `/Users/ws/Desktop/images`).

During launching `demo.py` you may initialize some parameters such as path to folder which include images for proccesing and path to folder in which results will be saved.

```
python demo.py -dd 'path_to_images' -rd 'path_to_results'
```
You should write:

```
python demo.py
```
for launching with test images, which are in folder `images`. 

During programs work you may add new files to the folder `path_to_images`. And programme will be processing them in real-time.

For stopping you need click Control-C. At the end of the programs work, the research results will be recorded in the info file `path_to_results/info.json`.