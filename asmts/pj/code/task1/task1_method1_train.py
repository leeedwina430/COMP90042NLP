#########################
# Method 1:  ANN + BERT #
#########################

METHOD = 1

import os
import json
import pandas as pd
import numpy as np

dir = ""
data_dir = dir + "data"
train_claim_path = os.path.join(data_dir, 'train-claims.json')
dev_claim_path = os.path.join(data_dir, 'dev-claims.json')
test_claim_path = os.path.join(data_dir, 'test-claims-unlabelled.json')
evidence_path = os.path.join(data_dir, 'evidence.json')
dev_baseline_path = os.path.join(data_dir, 'dev-claims-baseline.json')

train_claim = pd.DataFrame.from_dict(json.load(open(train_claim_path, 'r')), orient='index')
dev_claim = pd.DataFrame.from_dict(json.load(open(dev_claim_path, 'r')), orient='index')
test_claim = pd.DataFrame.from_dict(json.load(open(test_claim_path, 'r')), orient='index')
evidence = pd.DataFrame.from_dict(json.load(open(evidence_path, 'r')), orient='index')
dev_baseline = pd.DataFrame.from_dict(json.load(open(dev_baseline_path, 'r')), orient='index')

train_claim['num_index'] = np.arange(len(train_claim))
dev_claim['num_index'] = np.arange(len(dev_claim))
test_claim['num_index'] = np.arange(len(test_claim))
evidence['num_index'] = np.arange(len(evidence))

print("data loaded successfully!")
#%%
import time
from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation

# prepare data
train = train_claim.iloc[:int(len(train_claim)*0.8)]
val = train_claim.iloc[int(len(train_claim)*0.8):]

POS_SCORE = 0.95
NEG_SCORE = 0.5

start = time.time()
train_examples = []
for i in range(len(train)):
    for j in range(len(train['evidences'][i])):
        train_examples.append(InputExample(texts=[train['claim_text'][i], evidence[0][train['evidences'][i][j]]], label=POS_SCORE))
    for j in range(100):
        cur_evidence_idx = evidence.index[np.random.randint(len(evidence))]
        while cur_evidence_idx in train['evidences'][i]:
            cur_evidence_idx = evidence.index[np.random.randint(len(evidence))]
        train_examples.append(InputExample(texts=[train['claim_text'][i], evidence[0][cur_evidence_idx]], label=NEG_SCORE))

print('preprocess training data: ', time.time() - start, ' s')
print(len(train_examples))


start = time.time()
sentences1, sentences2, scores = [], [], []
for i in range(len(val)):
    for j in range(len(val['evidences'][i])):
        sentences1.append(val['claim_text'][i])
        sentences2.append(evidence[0][val['evidences'][i][j]])
        scores.append(POS_SCORE)
    for j in range(10):
        cur_evidence_idx = evidence.index[np.random.randint(len(evidence))]
        while cur_evidence_idx in val['evidences'][i]:
            cur_evidence_idx = evidence.index[np.random.randint(len(evidence))]
        sentences1.append(val['claim_text'][i])
        sentences2.append(evidence[0][cur_evidence_idx])
        scores.append(NEG_SCORE)

print('preprocess validation data: ', time.time() - start, ' s')
print(len(sentences1))

#%%
from torch.utils.data import DataLoader
import torch

BATCH_SIZE = 128   
EPOCHS = 60

#Define the model and the train loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
word_embedding_model = models.Transformer('bert-base-cased', max_seq_length=256)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# dense_model = models.Dense(
#     in_features=pooling_model.get_sentence_embedding_dimension(),
#     out_features=128,
#     activation_function=nn.Tanh(),
# )
# model.add_module('dense', dense_model)
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])

model.to(device)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.CosineSimilarityLoss(model)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores, write_csv=True)

#Tune the model
# model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=EPOCHS, warmup_steps=100)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=100,
    evaluator=evaluator,
    evaluation_steps=1000,
    show_progress_bar=True,
    output_path="output",
    checkpoint_save_steps=10000,
    save_best_model=True,
    checkpoint_path=f'output/checkpoint-{METHOD}'
)

print('finished training!')

# save model
model.save(dir + f'model/claim_evidence_bert_cased_{EPOCHS}_100')
print('model saved!')