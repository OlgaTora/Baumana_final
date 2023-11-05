from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score

from dags.logger import logger


class LinearModel:

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X, need_proba=False):
        if need_proba:
            return self.model.predict_proba(X)[:, 1]
        if need_proba:
            return self.model.predict(X)

    def get_metrics(self, X_test, y_test):
        predict = self.model.predict(X_test)
        predict_proba = self.model.predict_proba(X_test)[:, 1]

        metrics = {"F1_score": f1_score(y_test, predict, pos_label='Cancelled'),
                   "Recall": recall_score(y_test, predict, pos_label='Cancelled'),
                   'Accuracy': accuracy_score(predict, y_test),
                   "Precision": precision_score(y_test, predict, pos_label='Cancelled'),
                   "ROC_AUC": roc_auc_score(y_test, predict_proba)}
        logger.info(metrics)
        return metrics
