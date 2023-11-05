from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, recall_score, precision_score, accuracy_score
from dags.logger import logger


class LogisticRegressionModel:

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def prediction(self, X, need_proba=False):
        if need_proba:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def get_metrics(self, X_test, y_test):
        y_pred = self.model.prediction(X_test)
        predict_proba = self.model.prediction(X_test, True)

        metrics = {'F1_score': f1_score(y_test, y_pred, pos_label='Cancelled'),
                   'Recall': recall_score(y_test, y_pred),
                   'Accuracy': accuracy_score(y_pred, y_test),
                   'Precision': precision_score(y_test, y_pred, pos_label='Cancelled'),
                   'ROC_AUC': roc_auc_score(y_test, predict_proba)}
        logger.info(metrics)
        return metrics
