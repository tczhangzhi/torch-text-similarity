# torch-text-similarity
Implementations of models and metrics for semantic text similarity. Includes fine-tuning and prediction of models. Thanks for the elegent implementations of @Andriy Mulyar, who has published a lot of useful codes.

# Installation

Install with pip:

```
pip install torch-text-similarity
```

# Use
Maps batches of sentence pairs to real-valued scores in the range [0,5]
```python
import torch

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

predictions = learner.predict([('The patient is sick.', 'Grass is green.'),
                               ('A prescription of acetaminophen 325 mg was given.', ' The patient was given Tylenol.')
                               ])

print(predictions)
```
Make submission to a semantic text similarity competition
```python
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
```
Ensemble two pre-trained models and make a submission
```python
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
```
More [examples](/examples).

# Installation

The data sets in the examples can be found here:

* [train.csv](https://github.com/tczhangzhi/torch-text-similarity/releases/download/v1.0.0-data/train.csv)
* [test.csv](https://github.com/tczhangzhi/torch-text-similarity/releases/download/v1.0.0-data/test.csv)
* [sts-train.csv](https://github.com/tczhangzhi/torch-text-similarity/releases/download/v1.0.0-data/sts-train.csv)
* [sts-test.csv](https://github.com/tczhangzhi/torch-text-similarity/releases/download/v1.0.0-data/sts-test.csv)
* [sts-dev.csv](https://github.com/tczhangzhi/torch-text-similarity/releases/download/v1.0.0-data/sts-dev.csv)
