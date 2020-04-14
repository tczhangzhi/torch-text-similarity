import torch
import pandas as pd

from torch_text_similarity import TextSimilarityLearner
from torch_text_similarity.data import train_eval_sts_a_dataset

learner = TextSimilarityLearner(batch_size=10,
                                model_name='web-bert-similarity',
                                loss_func=torch.nn.MSELoss(),
                                learning_rate=5e-5,
                                weight_decay=0,
                                device=torch.device('cuda:0'))

train_dataset, eval_dataset = train_eval_sts_a_dataset(learner.bert_tokenizer, path='./data/train.csv')

learner.load_train_data(train_dataset)
learner.train(epoch=1)

test_data = pd.read_csv('./data/test.csv')
preds_list = []
for i, row in test_data.iterrows():
    text_a = row['text_a']
    text_b = row['text_b']
    preds = learner.predict([(text_a, text_b)])[0]
    preds_list.append(preds)

submission = pd.DataFrame({"id": range(len(preds_list)), "label": preds_list})
submission.to_csv('./submission.csv', index=False, header=False)