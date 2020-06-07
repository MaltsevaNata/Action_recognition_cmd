from sklearn import svm
from utils.model import Model

class SVM(Model):
    def __init__(self, PCA = False, kernel="linear", C=65, gamma=0.001):
        super(SVM, self).__init__(PCA=PCA)
        self.model = svm.SVC(kernel=kernel, C=C, gamma=gamma, cache_size=9000)

