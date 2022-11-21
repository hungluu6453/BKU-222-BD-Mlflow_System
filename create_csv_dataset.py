import pandas as pd

from datasets import load_dataset


dataset = load_dataset("uit-nlp/vietnamese_students_feedback")

train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']

train_pd = pd.DataFrame.from_dict(train_dataset)
val_pd = pd.DataFrame.from_dict(train_dataset)
test_pd = pd.DataFrame.from_dict(train_dataset)

df = pd.concat([train_pd, val_pd, test_pd]).drop('topic', axis=1)
df.to_csv('vi_student_feedbacks.csv', index=False)