import os
import pickle
from ActionRecognitionClass import ActionRecognition

classes_names = []
print("To train a model on your videos you should place videos with common action to one folder and name this folder by action")
paths_to_actions = ['/home/natalia/Рабочий стол/all_train_data/Move_scaner/', '/home/natalia/Рабочий стол/all_train_data/Tune_angle/',
                    '/home/natalia/Рабочий стол/all_train_data/Tune_height/']

filename = 'models/svm_old' + '.sav'
modelname = 'svm'
PCA = False
pca_filename = None
for num in range(len(paths_to_actions)):
    classes_names.append(os.path.split(paths_to_actions[num])[0].split('/')[-1])
with open(filename[:-4]+'_classes.txt', 'wb') as file:
    pickle.dump(classes_names, file)
print("Creating objects...")
act_rec = ActionRecognition(PCA=PCA, model_file=filename, PCA_file=pca_filename)
print("Starting training...")
act_rec.train(paths_to_actions, classes_names, modelname)
