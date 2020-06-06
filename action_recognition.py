import argparse
import os
from ActionRecognitionClass import ActionRecognition
from termcolor import colored
import pickle

# Define the program description
help = 'To train a model on your videos you should place videos with common action to one folder. Then use --train to start'

# Initiate the parser with a description
parser = argparse.ArgumentParser(description=help)
parser.add_argument("--train", "-t", help="start training model on new data", action="store_true")
parser.add_argument("--predict", "-p", help="predict action on video using trained model", action="store_true")

args = parser.parse_args()

paths_to_actions = []
model_names = ['svm', 'kNN', 'decision_tree']
classes_names = []

def get_train_data():
    print("Enter path to videos of the {} action or enter 'n'".format(len(paths_to_actions) + 1))
    answer = input()
    if answer != 'n':
        if not os.path.exists(answer):
            print(colored("No such directory. Try again or enter 'n'", 'red'))
        else:
            paths_to_actions.append(answer)
            print(colored("Added path to action {} :{}".format(len(paths_to_actions), answer), 'green'))
            get_train_data()
    return

if args.train:
    print("To train a model on your videos you should place videos with common action to one folder and name this folder by action")
    get_train_data()
    if len(paths_to_actions) < 2:
        print(colored("You need >= 2 actions to train. Do you want to add more? y/n", 'red'))
        answer = input()
        if answer == 'n':
            print(colored("Not enough data to train", 'red'))
            exit(-1)
        else:
            get_train_data()
    print(colored("Got paths to actions: {}".format(paths_to_actions), 'green'))
    print("Enter filename to save your trained model (without extension)")
    filename = input() + '.sav'
    print("Enter model name to train. Choices: {}".format(model_names))
    modelname = input()
    while modelname not in model_names:
        print(colored("No such model. Try again. Choices: {}".format(model_names), 'red'))
        modelname = input()
    print("Do you want to use PCA before training? y/n")
    PCA = False
    pca_filename = None
    if input()=='y':
        pca_filename = filename[:-4]+'_PCA.txt'
        print("Your PCA matrix will be saved to file {}".format(pca_filename))
        PCA = True
    for num in range(len(paths_to_actions)):
        classes_names.append(os.path.splitext("path_to_file")[0])
    with open(filename[:-4]+'_classes.txt', 'wb') as file:
        pickle.dump(classes_names, file)
    print("Creating objects...")
    act_rec = ActionRecognition(PCA=PCA, model_file=filename, PCA_file=pca_filename)
    print("Starting training...")
    act_rec.train(paths_to_actions, classes_names, modelname)
    print(colored("Model is trained and saved to {}. Now you can make predictions on new data".format(filename), 'green'))

elif args.predict:
    print("Enter name of the trained model. Choices: {}".format(model_names))
    modelname = input()
    while modelname not in model_names:
        print(colored("No such model. Try again. Choices: {}".format(model_names), 'red'))
        modelname = input()
    print("Enter filename of the trained model (without extension)")
    filename = input()+'.sav'
    while not os.path.exists(filename):
        print(colored("No such directory. Try again or enter 'n'", 'red'))
        filename = input()+'.sav'
    pca_filename = filename[:-4] + '_PCA.txt'
    if os.path.exists(pca_filename):
        PCA = True
    else:
        pca_filename = None
        PCA = False
    with open(filename[:-4] + '_classes.txt', 'rb') as file:
        classes_names = pickle.load(file)
    act_rec = ActionRecognition(PCA=PCA, model_file=filename, PCA_file=pca_filename)
    act_rec.load_trained(modelname, actions_names=classes_names)
    print("Enter path to the video you want to predict")
    predict_file = input()
    act_rec.predict(predict_file)