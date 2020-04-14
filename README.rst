torch-text-similarity
=====================

Implementations of models and metrics for semantic text similarity.
Includes fine-tuning and prediction of models. Thanks for the elegent
implementations of @Andriy Mulyar, who has published a lot of useful
codes.

Installation
============

Install with pip:

::

    pip install torch-text-similarity

Use
===

Maps batches of sentence pairs to real-valued scores in the range [0,5]

.. code:: python

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

Make submission to a semantic text similarity competition

.. code:: python

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

    train_dataset, eval_dataset = train_eval_sts_a_dataset(learner.bert_tokenizer, path='/home/temp/Data/kaggle/data/train.csv')

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

More `examples </examples>`__.

Installation
============

The data sets in the examples can be found in Google Cloud Drive:

-  `train.csv <https://drive.google.com/open?id=1-qqNudszBOboQHNvQwHp6-hyBPxjGH1I>`__
-  `test.csv <https://drive.google.com/open?id=1Ph8F0d-JE61MAQicKx24GK29hRXciws9>`__
-  `sts-train.csv <https://drive.google.com/open?id=1BJCDxzKZDyvxzdtTFBP-gQzcZWmwClGX>`__
-  `sts-test.csv <https://drive.google.com/open?id=1NGrIg3DnbSjl4uKciL9WsiCFzK8Q726X>`__
-  `sts-dev.csv <https://drive.google.com/open?id=1OZxOC4Y9XU-ZTXVf78_DPu9edTZaYFRX>`__

