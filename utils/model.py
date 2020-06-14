from sklearn.metrics import accuracy_score
import pickle
import numpy as np
from numpy.linalg import eig
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score
import seaborn as sns

class Model:
    def __init__(self, PCA = False):
        self.model = None
        self.PCA = PCA
        self.PCA_matrix = None

    def load_trained_model(self, model_file, PCA_file=None, PCA=False):
        self.model = pickle.load(open(model_file, 'rb'))
        if PCA:
            self.PCA_matrix = np.loadtxt(PCA_file, dtype=float)

    def train(self, X, Y, model_file, PCA_file = None, show=False):
        if self.PCA:
            M = np.mean(X, axis=0)
            C = X - M
            C = C.transpose()
            V = np.cov(C.astype(float))
            val, vect = eig(V)
            indexes = np.argsort(val)[::-1]
            sorted_values = val[indexes]
            if show:
                plt.bar(np.arange(len(sorted_values)), sorted_values)
                plt.show()
            contribs = []
            summa = sum(sorted_values)
            for val in sorted_values:
                contrib = (val/summa)*100
                if contrib >= 0.1:
                    contribs.append(contrib)
            components_num = len(contribs)
            sorted_vectors = vect[indexes]
            self.PCA_matrix = sorted_vectors.T[:, 0:components_num]
            np.savetxt(PCA_file, self.PCA_matrix, fmt='%f')
            X = np.dot(X, self.PCA_matrix)
        print("Training model...")
        self.model.fit(X, Y)
        pickle.dump(self.model, open(model_file, 'wb'))
        print("FINISHED training. Check on train data:")
        '''y_predict' = model.p'redict(X)
        print("FINISHED training. Check on train data: accuracy score : ")
        print(accuracy_score(Y, y_predict))'''



    def predict(self, X, Y=None):
        if self.PCA:
            X = np.dot(X, self.PCA_matrix)
        y_predict = self.model.predict(X)
        #print(y_predict)
        #print("Predicted: {}".format(y_predict))
        #Y = [ 2 for i in range(y_predict.shape[0]) ]
        #print(Y)
        if Y is not None:
            print("Comparing results to expected. Accuracy:")
            print(accuracy_score(Y, y_predict))
            print("Recall: {}".format(recall_score(Y, y_predict, average='weighted')))
            print("Precision: {}".format(precision_score(Y, y_predict, average='weighted')))
            cf_matrix = confusion_matrix(Y, y_predict)
            print(cf_matrix)
            ax = sns.heatmap(cf_matrix, annot=True, cmap="Blues", cbar_kws={'label': 'Scale'}, fmt='g')
            ax.set(ylabel="True Label", xlabel="Predicted Label")
            plt.show()
        counts = np.bincount(y_predict)
        predicted = {}
        classes_num = len(np.unique(y_predict))
        for cl_num in range(classes_num):
            confidence = 100*np.max(counts[cl_num])/np.sum(counts)
            predicted[cl_num] = confidence
        return predicted

    def predict_class(self, X, Y=None):
        if self.PCA:
            X = np.dot(X, self.PCA_matrix)
        y_predict = self.model.predict(X)
        return y_predict