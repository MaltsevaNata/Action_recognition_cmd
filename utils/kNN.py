from sklearn.neighbors import KNeighborsClassifier
from utils.model import Model


class kNN(Model):
    def __init__(self, PCA=False, n_neighbors=3):
        super(kNN, self).__init__(PCA=PCA)
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
