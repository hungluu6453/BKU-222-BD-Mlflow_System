import mlflow
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

from data_preprocessing import create_pyspark_dataframe, clean_data

class Feedbacks_Classifier:
    def __init__(self) -> None:
        self.define_model()
        self.define_data()
        self.label_dict = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        self.eval_model()

    def define_model(self):
        self.model =  LogisticRegression(random_state=50, max_iter = 200)
    
    def define_data(self):
        sparkdf = create_pyspark_dataframe()
        cleaning_func = udf(lambda x : clean_data(x), StringType())
        df = sparkdf.withColumn('sentence', cleaning_func(col('sentence'))).toPandas()

        X = df['sentence']
        y = df['sentiment']

        self.count_vect = CountVectorizer()

        X_train_count = self.count_vect.fit_transform(X)
        self.tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_count)
        X_train_tf = self.tf_transformer.transform(X_train_count)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_tf, y, test_size=0.25, random_state=16)

    def eval_model(self):
        mlflow.sklearn.autolog()
        print(self.model)
        # Training
        self.model.fit(self.X_train,self.y_train)
        # Predicting based in X_test
        y_pred = self.model.predict(self.X_test)
        # Model Evaluating 
        Accuracy = round(accuracy_score(self.y_test, y_pred),3)
        Precision = round(precision_score(self.y_test, y_pred, average = "weighted"),3)
        Recall= round(recall_score(self.y_test, y_pred, average = "weighted"),3)
        F1=  round(f1_score(self.y_test, y_pred, average = "weighted"),3)    
        
        dict={'Ac':Accuracy, 'Pc':Precision, 'Rcll':Recall, 'F1':F1}
        print(dict)
        return(dict)

    def predict_feedbacks_type(self, text):
        X_test_count = self.count_vect.transform(text)
        X_test_tfidf = self.tf_transformer.transform(X_test_count)
        test_label = self.model.predict(X_test_tfidf)

        autolog_run = mlflow.last_active_run()
        return self.label_dict[test_label[0]]


if __name__ == "__main__":
    feedbacks_classifier = Feedbacks_Classifier()
    test_data = ['Thầy dạy hay lắm !!!']
    print(feedbacks_classifier.predict_feedbacks_type(test_data))