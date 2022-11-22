import mlflow

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from run import Feedbacks_Classifier
from ultilities.data_processing import push_unlabeled_data

app = FastAPI()

server_uri = 'http://127.0.0.1:5000'
mlflow.set_tracking_uri(server_uri)
model = Feedbacks_Classifier(load_from_registry=True, model_name='Feedback_Anlayzer', stage='Production')

class Request_Item(BaseModel):
    text: str

class Response_Item(BaseModel):
    text: str
    execution_time: float


@app.post("/api/v1/feedback_analyzer")
def analyze(Request: Request_Item):
    feedback = Request.text

    if feedback is None:
        return {'error': 'Please input a feedback'}
    
    sentiment, execution_time = model.predict_from_server(feedback)
    push_unlabeled_data(sentiment)

    return Response_Item(text = sentiment, execution_time= round(execution_time, 4))

if __name__ == '__main__':
    uvicorn.run("serve:app",host='0.0.0.0', port=8080)