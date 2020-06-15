import pickle
from ActionRecognitionClass import ActionRecognition
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

modelname = 'decision_tree'
filename = '/home/natalia/PycharmProjects/action_recognition/models/tree_new_10.sav'
with open(filename[:-4] + '_classes.txt', 'rb') as file:
    classes_names = pickle.load(file)
act_rec = ActionRecognition(PCA=False, model_file=filename, PCA_file=None)
act_rec.load_trained(modelname, actions_names=classes_names)
predicted_files = ['/home/natalia/Рабочий стол/all_train_data/Move_scaner5.mp4','/home/natalia/Рабочий стол/all_train_data/Tune_angle2.mp4',
                   '/home/natalia/Рабочий стол/all_train_data/Tune_height2.mp4']
y_predict = []
Y = []
cl = 0
for predict_file in predicted_files:
    y = act_rec.predict_class_by_video_descriptor(predict_file)
    y_predict=*y_predict, *y
    Y = *Y, *[cl for i in range(y.shape[0])]
    cl+=1
print("Comparing results to expected. Accuracy:")
print(accuracy_score(Y, y_predict))
print("Recall: {}".format(recall_score(Y, y_predict, average='weighted')))
print("Precision: {}".format(precision_score(Y, y_predict, average='weighted')))
cf_matrix = confusion_matrix(Y, y_predict)
print(cf_matrix)
ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues", cbar_kws={'label': 'Scale'}, fmt='g')
ax.set(ylabel="True Label", xlabel="Predicted Label")
plt.show()