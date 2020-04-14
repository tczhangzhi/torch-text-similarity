import torch
import pandas as pd

from torch_text_similarity import TextSimilarityLearner
from torch_text_similarity.data import train_eval_sts_a_dataset

learner_1 = TextSimilarityLearner(model_name='web-bert-similarity', device=torch.device('cuda:0'))
learner_2 = TextSimilarityLearner(model_name='clinical-bert-similarity', device=torch.device('cuda:0'))


test_data = pd.read_csv('./data/test.csv')
preds_list = []
for i, row in test_data.iterrows():
    text_a = row['text_a']
    text_b = row['text_b']
    preds = learner_1.predict([(text_a, text_b)])[0] * 0.55 + learner_2.predict([(text_a, text_b)])[0] * 0.45
    preds_list.append(preds)

submission = pd.DataFrame({"id": range(len(preds_list)), "label": preds_list})
submission.to_csv('./submission.csv', index=False, header=False)