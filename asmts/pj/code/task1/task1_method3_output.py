###########################
# Method 3: Siamese BERT  #
###########################

METHOD = 3
EPOCHS = 50

import os
import json
import pandas as pd
import numpy as np

dir = "../../"
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
evi_idx = list(evidence.index)

print("data loaded successfully!")

# #%%
# evaluate (real)
# load model
from sentence_transformers import SentenceTransformer, util
import torch

model = SentenceTransformer(f'model/claim_evidence_bert_{METHOD}_cased_{EPOCHS}')
model.eval()
print('model loaded!')

BATCH_SIZE_EVAL = 2048
CUT_OFF = 5
THRESHOLD = None

# embed data
claim_embedding_eval = model.encode(dev_claim['claim_text'], convert_to_tensor=True, 
                                    show_progress_bar=True)
torch.save(claim_embedding_eval, 'output/dev_embeddings_eval.pt')
print('claim embedding finished!')


evidence_embedding_eval = model.encode(evidence[0], batch_size=BATCH_SIZE_EVAL,
                                  convert_to_tensor=True, show_progress_bar=True)
torch.save(evidence_embedding_eval, 'output/evi_embeddings_eval.pt')
print('evidence embedding finished!')

# usage
# claim_embedding_eval = torch.load('output/dev_embeddings_eval.pt')
# evidence_embedding_eval = torch.load('output/evi_embeddings_eval.pt')

#%%
# # normalize???
# from sklearn.preprocessing import normalize
# evidence_embedding_eval = normalize(evidence_embedding_eval)
# claim_embedding_eval = normalize(claim_embedding_eval)

#%%
import time

start = time.time()
dev_output_bf = {}

# Calculate cosine similarity between the batch of claims and evidence
similarities = util.pytorch_cos_sim(claim_embedding_eval, evidence_embedding_eval)
print('similarity finished!')

k = CUT_OFF

# Iterate over the claims in the current batch
for j, text_id in enumerate(dev_claim.index):
    dev_output_bf[text_id] = {}
    cur_nearest_neighbors = []
    cur_indices = []
    cur_similarity = []

    top_k_indices = similarities[j].argsort(descending=True)[:k]

    top_k_evidence = [evi_idx[i] for i in top_k_indices]
    dev_output_bf[text_id]['evidences'] = top_k_evidence

# print(f'{len(dev_claim)} queries / {len(evidence)} evidence\n brutal force : ', time.time()-start, ' s')
print(f'threshold = {THRESHOLD}, cut-off = {CUT_OFF}, \nbatch : ', time.time()-start, ' s')

len_counts = []
for key in dev_output_bf:
    len_counts.append(len(dev_output_bf[key]['evidences']))
print("average evidences: ", sum(len_counts)/len(dev_output_bf))

# save output
for text_id in dev_output_bf:
    dev_output_bf[text_id]['claim_label'] = 'SUPPORTS'
    # dev_output_bf[text_id].pop('similarities')
with open(f"output/SBERT_{METHOD}_cased_{EPOCHS}.json", "w") as out_f:
    json.dump(dev_output_bf, out_f)

print('finished!')
