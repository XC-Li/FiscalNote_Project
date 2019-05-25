from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    confusion_matrix, log_loss, classification_report, precision_recall_curve
# from util_code.corpus_loader import corpus_loader
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd


class Pipeline(object):
    """
    The one-stop pipeline for training, evaluating and diagnosing any sklearn model
    Designed by: Xiaochi (George) Li github.com/XC-Li
    """
    def __init__(self, x, y, vectorizer, model, silent=False, sampler=None):
        """
        define the components of the pipeline
        """
        self.vectorizer = vectorizer
        self.model = model
        self.sampler = sampler
        self.X = x
        self.y = y
        self.X_train = self.X_dev = self.y_train = self.y_dev = self.y_pred = None
        self.silent = silent
        self.y_train_pred = None
        self.X_train_processed = self.X_dev_processed = None
        self.X_train_vectorized = self.X_dev_vectorized = None
        self.probability = None

    def print_(self, message, end=''):
        if self.silent is False:
            print(message, end=end)

    def split(self):
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_dev, self.y_train, self.y_dev = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42)

    def preprocess(self):
        """
        preprocess
        """
        self.print_('preprocess -> ', end='')
        self.X_train_processed = self.X_train
        self.X_dev_processed = self.X_dev

    def vectorization(self):
        """
        vectorization
        """
        start_time = timer()
        self.print_('vectorization ', end='')
        self.vectorizer.fit(self.X_train_processed)
        self.X_train_vectorized = self.vectorizer.transform(self.X_train_processed)
        self.X_dev_vectorized = self.vectorizer.transform(self.X_dev_processed)
        end_time = timer()
        self.print_('(Time:' + str(round(end_time - start_time)) + 's) ->', end='')

    def resampling(self):
        if self.sampler is not None:
            self.print_('Resampling ->', end='')
            self.X_train_vectorized, self.y_train = self.sampler.fit_sample(self.X_train_vectorized, self.y_train)
        else:
            self.print_('No resampling ->', end='')

    def train_model(self):
        """
        train model
        """
        start_time = timer()
        self.print_('train model ', end='')
        self.model.fit(self.X_train_vectorized, self.y_train)
        end_time = timer()
        self.print_('(Time:' + str(round(end_time - start_time)) + 's) -> ', end='')

    def model_evaluation(self):
        """
        model evaluation
        """
        self.print_('model evaluation', end='\n')
        self.y_pred = self.model.predict(self.X_dev_vectorized)
        self.probability = self.model.predict_proba(self.X_dev_vectorized)
        self.y_train_pred = self.model.predict(self.X_train_vectorized)
        if not self.silent:  # turn off any print
            print('-----------Metrics on Dev Set----------------------------')
            print('Log Loss:', log_loss(self.y_dev, self.y_pred))
            print(classification_report(self.y_dev, self.y_pred))  # one classification report can replace others
            print('-----------Metrics on Training Set-----------------------')
            print('Log Loss:', log_loss(self.y_train, self.y_train_pred))
            print(classification_report(self.y_train, self.y_train_pred))

            precision, recall, _ = precision_recall_curve(self.y_dev, self.probability[:, 1])
            # print('F1 score:', f1_score(self.y_dev, self.y_pred))
            # print('Accuracy:', accuracy_score(self.y_dev, self.y_pred))
            # print('Precision:', precision_score(self.y_dev, self.y_pred))
            # print('Recall:', recall_score(self.y_dev, self.y_pred))
            confusion_matrix_ = confusion_matrix(self.y_dev, self.y_pred, labels=[1, -1])
            print('Confusion Matrix:\n      Prediction\n        1   -1')
            print('True 1', end='')
            print(confusion_matrix_[0])
            print('    -1', end='')
            print(confusion_matrix_[1])
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.show()

    def get_f1(self):
        """
        get the f1 score for train and test set
        :return: Dict
        """
        self.y_pred = self.model.predict(self.X_dev_vectorized)
        self.probability = self.model.predict_proba(self.X_dev_vectorized)
        self.y_train_pred = self.model.predict(self.X_train_vectorized)
        test_report = classification_report(self.y_dev, self.y_pred, output_dict=True)
        test_f1_neg = test_report['-1']['f1-score']
        test_f1_pos = test_report['1']['f1-score']
        train_report = classification_report(self.y_train, self.y_train_pred, output_dict=True)
        train_f1_neg = train_report['-1']['f1-score']
        train_f1_pos = train_report['1']['f1-score']
        self.print_('model evaluation', end='\n')
        return {'test_f1_neg': test_f1_neg, 'test_f1_pos': test_f1_pos,
                'train_f1_neg': train_f1_neg, 'train_f1_pos': train_f1_pos}

    def exec(self, silent=False):
        """
        run pipeline
        """
        self.silent = silent
        self.split()
        self.preprocess()
        self.vectorization()
        self.resampling()
        self.train_model()
        if silent is False:
            self.model_evaluation()
            return self.model
        else:
            return self.get_f1()


    def test_sample_analysis(self):
        """
        combine the speech in test set with the predicted probability from the model
        can be used to analysis the extreme wrong classification samples
        Returns:
            df_pred(Pandas DataFrame):
        """
        df_pred = pd.DataFrame({'speech': self.X_dev, 'TrueY': self.y_dev, 'PredY': self.y_pred,
                                'P(y=-1)': self.probability[:, 0], 'P(y=1)': self.probability[:, 1]})
        return df_pred

    def _confusion_matrix_analysis(self, df_pred, print_=False):
        """
        Hidden funtion: split the predited test set into a confusion matrix for plotting
        Args:
            df_pred(pd.DataFrame)
            print_(Bool)
        Returns:
            dict
        """
        positive = df_pred.loc[df_pred['TrueY'] == 1]
        negative = df_pred.loc[df_pred['TrueY'] == -1]
        true_positive = df_pred.loc[(df_pred['TrueY'] == 1) & (df_pred['PredY'] == 1)]
        false_positive = df_pred.loc[(df_pred['TrueY'] == -1) & (df_pred['PredY'] == 1)]
        false_negative = df_pred.loc[(df_pred['TrueY'] == 1) & (df_pred['PredY'] == -1)]
        true_negative = df_pred.loc[(df_pred['TrueY'] == -1) & (df_pred['PredY'] == -1)]
        if print_:
            print('Total:', df_pred.shape[0])
            print('Positive:', positive.shape[0])
            print('Negative:', negative.shape[0])
            print('True Positive:', true_positive.shape[0])
            print('False Positive:', false_positive.shape[0])
            print('False Negative:', false_negative.shape[0])
            print('True Negative:', true_negative.shape[0])
        if (true_positive.shape[0] + false_positive.shape[0]) > 0:
            tp_rate = true_positive.shape[0] / (true_positive.shape[0] + false_positive.shape[0])
            if print_:
                print('[TP/(TP+FP)]:', round(tp_rate, 4))
        else:
            tp_rate = -1
        if (true_negative.shape[0] + false_negative.shape[0]) > 0:
            tn_rate = true_negative.shape[0] / (true_negative.shape[0] + false_negative.shape[0])
            if print_:
                print('[TN/(TN+FN)]:', round(tn_rate, 4))
        else:
            tn_rate = -1
        return ({'total': df_pred.shape[0], 'positive': positive, 'negative': negative,
                 'true_positive': true_positive, 'false_positive': false_positive,
                 'false_negative': false_negative, 'true_negative': true_negative,
                 'tp_rate': tp_rate, 'tn_rate': tn_rate})

    def threshold_graph(self):
        """
        plot the precision and retention when the threshold changes
        """
        df_pred = self.test_sample_analysis()
        prediction = self._confusion_matrix_analysis(df_pred)
        threshold_table = []
        true_positive_total = prediction['true_positive'].shape[0]
        true_negative_total = prediction['true_negative'].shape[0]
        for probability_threshold in [i / 100 for i in range(50, 100, 1)]:
            #     probability_threshold = 0.9
            strong_positive = df_pred.loc[df_pred['P(y=1)'] > probability_threshold]
            sp = self._confusion_matrix_analysis(strong_positive)
            sp_total = sp['total']  # / true_positive_total
            sp_tpr = sp['tp_rate']
            strong_negative = df_pred.loc[df_pred['P(y=-1)'] > probability_threshold]
            sn = self._confusion_matrix_analysis(strong_negative)
            sn_total = sn['total']  # / true_negative_total
            sn_tnr = sn['tn_rate']
            threshold_table.append([probability_threshold, sp_total, sp_tpr, sn_total, sn_tnr])
        threshold_df = pd.DataFrame(threshold_table)
        threshold_df.columns = ['threshold', 'positive_retention', 'true_positive_rate', 'negative_retention',
                                'true_negative_rate']
        threshold_df.set_index('threshold')
        threshold_df.plot.line(x='threshold', y='true_positive_rate')
        threshold_df.plot.line(x='threshold', y='positive_retention')
        threshold_df.plot.line(x='threshold', y='true_negative_rate')
        threshold_df.plot.line(x='threshold', y='negative_retention')
        plt.show()


# if __name__ == '__main__':  # sample use
#     from sklearn.feature_extraction.text import CountVectorizer
#     from sklearn.svm import SVC
#     df = corpus_loader()
#     X = df['text']
#     y = df['support']
#     convote_pipeline = Pipeline(X, y, CountVectorizer(binary=True), SVC(kernel='linear', random_state=42))
#     print(convote_pipeline.exec())
