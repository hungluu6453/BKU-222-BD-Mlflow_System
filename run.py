import mlflow
import mlflow.pyfunc

import time

from sklearn.feature_extraction.text import  TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ultilities.data_processing import load_cleaned_data, clean_data

server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(server_uri)

class Feedbacks_Classifier:
    def __init__(self, load_from_registry = None, model_name = None, stage = None) -> None:
        self.define_data()
        self.label_dict = { 0: 'Negative', 
                            1: 'Neutral', 
                            2: 'Positive'
                            }
        self.model_name_list = ['logistic-regression',
                                'random-forest',
                                'knn',
                                'SVM'
                                ]
        self.experiment_id = '0'
        
        if load_from_registry and model_name and stage:
            self.model = mlflow.pyfunc.load_model(
                    model_uri=f"models:/{model_name}/{stage}"
                )
            
    def define_model(self, model_name = 'random-forest'):
        if model_name == 'logistic-regression':
            return  LogisticRegression()
        elif model_name == 'random-forest':
            return RandomForestClassifier()
        elif model_name == 'knn':
            return KNeighborsClassifier()
        elif model_name == 'SVM':
            return SVC()
        else:
            pass
            
    def define_data(self):
        df = load_cleaned_data()

        X = df['sentence']
        y = df['sentiment']

        self.tfidf_vect = TfidfVectorizer(use_idf=True)
        X_train_tfidf = self.tfidf_vect.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_train_tfidf, y, test_size=0.25, random_state=16)

    def run_single_model(self, model_name, run_name = None):
        mlflow.sklearn.autolog()

        self.model = self.define_model(model_name)

        with mlflow.start_run(run_name=run_name, experiment_id=self.experiment_id):
            self.model.fit(self.X_train,self.y_train)
            self.model.predict(self.X_test)

    def run_all_models(self, experiment_name):
        self.experiment_id = mlflow.create_experiment(name=experiment_name)

        for model_name in self.model_name_list:
            self.run_single_model(model_name)

    def predict_from_local(self, text):
        text = clean_data(text)
        X_test_tfidf = self.tfidf_vect.transform(text)
        test_label = self.model.predict(X_test_tfidf)

        return self.label_dict[test_label[0]]

    def predict_from_server(self, text):
        start_time = time.time()
        #text = clean_data(text)
        X_test_tfidf = self.tfidf_vect.transform([text])
        test_label = self.model.predict(X_test_tfidf)
        return self.label_dict[test_label[0]], time.time() - start_time

if __name__ == "__main__":
    feedbacks_classifier = Feedbacks_Classifier(load_from_registry=True, model_name='Feedback_Anlayzer', stage='Production')

    
    #feedbacks_classifier.run_single_model('random-forest')

    #feedbacks_classifier.run_all_models('test_all_model_on_data_22112022')

    test_data = ['Thầy dạy hay lắm !!!']
    #print(feedbacks_classifier.predict_from_local(test_data))
    print(feedbacks_classifier.predict_from_server(test_data))