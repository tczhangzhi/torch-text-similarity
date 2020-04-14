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