# Action recognition Command Line Interface (CLI)
## Tool description
  The aim of the work is to create a simple user interface to interact with machine learning methods of human actions evaluation based on skeleton poses provided by Openpose. Several methods for solving the problem are selected: SVM, kNN, decision tree. Training dataset was created to conduct the research and test the tool. Processing algorithms are developed and experimental studies are carried out, which showed that the selected methods are applicable for the task of evalusting actions, but require refinement and expansion of the dataset.
 ### Tool functionality
  --help , -h    show this help message and exit\
  --train, -t    start training model on new data, PCA available\
  --predict, -p  predict action on video using trained model, PCA available\
  Optional parameter PCA serves for to decrease number of training parameters in the model.\
   ***For now only SVM method is available. Update soon.***
## Installation
   **1. Install OpenPose**  https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/installation.md \
    Required:\
    CUDA (Nvidia GPU) version:\
    - NVIDIA graphics card with at least 1.6 GB available (the nvidia-smi command checks the available GPU memory in Ubuntu).\
    - At least 2.5 GB of free RAM memory for BODY_25 model or 2 GB for COCO model (assuming cuDNN installed).\
    - Highly recommended: cuDNN. \
   **2. Install dependencies** from requirements.txt
## Define input data
  Input data to train models and predict is folders with videos *.mp4. Every video contains basically 1 action done by 1 person. Every folder contains videos with one action and is named by action name. Example: If you want to train the model to recognize 3 classes, you need to create 3 folders. 
  1. project_path/Move_scaner/ stores videos like this:  \
    ![Video1](https://media.giphy.com/media/cgeVZMM88qWlj6Nzzf/giphy.gif)
  2. project_path/Tune_height/ with videos:  \
    ![Video2](https://media.giphy.com/media/LME1WK8M6zMGU6exuN/giphy.gif)
  3. project_path/Tune_angle/ with videos:\
    ![Video3](https://media.giphy.com/media/RLE8FhEeXSYN5zAp71/giphy.gif)
## Train model example

> ~/PycharmProjects/action_recognition$ python3 action_recognition.py -t \
To train a model on your videos you should place videos with common action to one folder and name this folder by action \
Enter path to videos of the 1 action or enter 'n' \
/home/Move_scaner/ \
Added path to action 1 :/home/Move_scaner/ \
>Enter path to videos of the 2 action or enter 'n' \
/home/Tune_angle/ \
Added path to action 2 :/home/Tune_angle/ \

>Enter path to videos of the 3 action or enter 'n' \
/home/Tune_height/ \
Added path to action 3 :/home/Tune_height/ \
>Enter path to videos of the 4 action or enter 'n' \
n \
>Got paths to actions: ['/home/Move_scaner/', '/home/Tune_angle/', '/home/Tune_height/'] \

>Enter filename to save your trained model (without extension) \
models/check_training \

>Enter model name to train. Choices: ['svm', 'kNN', 'decision_tree'] \
svm \

>Do you want to use PCA before training? y/n \
n \
>Creating objects... \
Starting OpenPose Python Wrapper... \
Auto-detecting all available GPUs... Detected 1 GPU(s), using 1 of them starting at GPU 0. \
Starting training...  
Training model... \
FINISHED training. Check on train data: \
Comparing results to expected. 
Accuracy: 1.0 \
Recall: 1.0 \
Precision: 1.0 \
[[11  0  0]
 [ 0  6  0]
 [ 0  0 22]] \
>Model is trained and saved to models/check_training.sav. Now you can make predictions on new data 


