from sklearn.tree import DecisionTreeClassifier
from utils.model import Model

class decision_tree(Model):
    def __init__(self, PCA=False, criterion="entropy", max_depth=4, min_samples_split=10, min_samples_leaf = 3):
        super(decision_tree, self).__init__(PCA=PCA)
        self.model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf = min_samples_leaf)
