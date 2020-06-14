from utils.recognize_pose import create_pose_json
from utils.get_images import get_images
from utils.start_openpose import start_openpose
import json
import sklearn.model_selection as model_selection
import utils.SVM as SVM
import utils.kNN as kNN
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import pandas as pd
from datetime import datetime

class ActionRecognition:
    def __init__(self, PCA, model_file, PCA_file = None):
        self.PCA = PCA
        self.trained = False
        self.model = None
        self.model_file = model_file
        self.PCA_file = PCA_file
        self.op, self.opWrapper = start_openpose()
        self.action_names = []

    def process_image(self, image):
        data = []
        rassty = 0
        nose = 0
        rankle = 0
        lankle = 0
        pose = create_pose_json(self.op, self.opWrapper, image)  # includes changing koords to relative to neck
        pose = json.loads(pose)['Person0']
        for part in pose:
            if part['bodypart'] == 'Nose':
                nose = part['Y']
            elif part['bodypart'] == 'RAnkle':
                rankle = part['Y']
            elif part['bodypart'] == 'LAnkle':
                lankle = part['Y']
        rassty = max(abs(float(nose) - float(lankle)), abs(float(nose) - float(rankle)))  # find man's height
        # normalization by height
        for part in pose:
            if rassty == 0:
                rassty = 1
            data.append(float(part['X']) / rassty)
            data.append(float(part['Y']) / rassty)
        return data  # 50 items list

    def get_data_from_video(self, video):
        data = []
        if not video.endswith('.mp4'):
            return
        poses = []
        images = get_images(video)
        for image in images:
            data.append([])
            image_data = self.process_image(image)
            data[-1] = image_data
        return data

    def load_trained(self, model_name, actions_names):
        self.action_names = actions_names
        if model_name=='svm':
            svm = SVM.SVM(PCA=self.PCA)
            self.model = svm
            svm.load_trained_model(model_file=self.model_file, PCA=self.PCA, PCA_file=self.PCA_file)
        elif model_name=='kNN':
            knn = kNN.kNN(PCA=self.PCA)
            self.model = knn
            knn.load_trained_model(model_file=self.model_file, PCA=self.PCA, PCA_file=self.PCA_file)

    def train(self, paths_to_classes, action_names, model_name):
        self.action_names = action_names
        X = []
        Y = []
        actions_num = len(paths_to_classes)
        for action_num in range(actions_num):
            for roots, dirs, files in os.walk(paths_to_classes[action_num]):
                for video in files:
                    data = self.get_data_from_video(paths_to_classes[action_num] + video)
                    X = *X, *data
                    Y = *Y, *([action_num] * len(data))

        X = np.array(X).astype(float)
        Y = np.array(Y).astype(int)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, train_size=0.8, test_size=0.2,
                                                                            random_state=101)
        if model_name == 'svm':
            svm = SVM.SVM(PCA=self.PCA)
            self.model = svm
            svm.train(X_train, y_train, self.model_file, PCA_file=self.PCA_file)
            self.trained = True
            svm.predict(X_test, y_test)
        elif model_name == 'kNN':
            knn = kNN.kNN(PCA=self.PCA)
            self.model = knn
            knn.train(X_train, y_train, self.model_file, PCA_file=self.PCA_file)
            self.trained = True
            knn.predict(X_test, y_test)
        elif model_name == 'gmm':
            gmm = GaussianMixture(n_components=3)

            # Fit the GMM model for the dataset
            # which expresses the dataset as a
            # mixture of 3 Gaussian Distribution
            d = pd.DataFrame(X_train)
            gmm.fit(d)

            # Assign a label to each sample
            labels = gmm.predict(d)
            d['labels'] = labels
            d0 = d[d['labels'] == 0]
            d1 = d[d['labels'] == 1]
            d2 = d[d['labels'] == 2]
            f, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)

            '''import math
            distances = {}
            center1 = []
            center2 = []
            for i in range(len(d0)):
                for p1 in d[:,i]:
                    center1 += p1**2
                center1 = math.sqrt(center1)
                for p2 in d0[:,i]:
                    center2 += p2**2
                center2 = math.sqrt(center2)
                dist += (d0[i]-d[i])**2'''

            # plot three clusters in same plot
            ax1.scatter(d0[0], d0[1], c='r')
            ax1.scatter(d1[0], d1[1], c='yellow')
            ax1.scatter(d2[0], d2[1], c='g')
            ax1.set_xlabel('Predicted')
            ax2.scatter(d[0], d[1], c=y_train, cmap='brg')
            ax2.set_xlabel('Original')
            plt.show()
    def predict(self, video):
        X = self.get_data_from_video(video)
        predicted = self.model.predict(X)
        predicted_actions = {}
        for key in predicted.keys():
            action = self.action_names[key]
            predicted_actions[action] = predicted[key]
        print("Predicted: [action: confidence]: {}".format(predicted_actions))

    def predict_by_img(self, image):
        return self.model.predict_class(image)

    def real_time_prediction(self, video):
        cap = cv2.VideoCapture(video)
        while (cap.isOpened()):
            start_time = datetime.now()
            ret, frame = cap.read()
            if ret == True:
                frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                pose = np.array(self.process_image(frame)).astype(float)
                pose = np.reshape(pose,(1,50))
                predicted = self.predict_by_img(pose)[0]
                print(predicted)
                action = self.action_names[predicted]
                print(action)
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(frame, action, (int(frameWidth/2), 30), font, 1, color=(0, 255, 0), thickness=2)

                cv2.imshow('frame', frame)
                print("Time: {}".format(datetime.now() - start_time))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
