import os
import json
import pandas as pd
import numpy as np
import time
import torch
from sentence_transformers import SentenceTransformer, util
import faiss
import tqdm


dir = ""
data_dir = dir + "data"
train_claim_path = os.path.join(data_dir, 'train-claims.json')
dev_claim_path = os.path.join(data_dir, 'dev-claims.json')
test_claim_path = os.path.join(data_dir, 'test-claims-unlabelled.json')
evidence_path = os.path.join(data_dir, 'evidence.json')

train_data = json.load(open(train_claim_path, 'r'))
dev_data = json.load(open(dev_claim_path, 'r'))
test_data = json.load(open(test_claim_path, 'r'))
evidence_data = json.load(open(evidence_path, 'r'))

# print(len(train_data), len(dev_data), len(test_data), len(evidence_data))
# # print(dev_data)

# train_claim = pd.DataFrame.from_dict(json.load(open(train_claim_path, 'r')), orient='index')
# dev_claim = pd.DataFrame.from_dict(json.load(open(dev_claim_path, 'r')), orient='index')
# test_claim = pd.DataFrame.from_dict(json.load(open(test_claim_path, 'r')), orient='index')
# evidence = pd.DataFrame.from_dict(json.load(open(evidence_path, 'r')), orient='index')

# print(len(train_claim), len(dev_claim), len(test_claim), len(evidence))
print("data loaded successfully!")


names = ['train', 'dev', 'test', 'evidence']
batch_sizes = [512, 512, 512, 2048]

sentences1 = [train_data[item]['claim_text'] for item in train_data.keys()]
sentences2 = [dev_data[item]['claim_text'] for item in dev_data.keys()]
sentences3 = [test_data[item]['claim_text'] for item in test_data.keys()]
sentences4 = [evidence_data[item] for item in evidence_data.keys()]
sentences = [sentences1 , sentences2 , sentences3 , sentences4]

print(len(sentences[0]),len(sentences[1]),len(sentences[2]),len(sentences[3]))

print("sentences loaded successfully!")


# Set the device for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentenceTransformer('bert-base-cased').to(device)

for name, sentence, batch_size in zip(names, sentences, batch_sizes):
    # Define the sentences
    num_batches = len(sentence) // batch_size + 1  # 如果加一，batch_size不能整除

    start = time.time()
    embeddings = np.empty((len(sentence), model.get_sentence_embedding_dimension()))
    for i in range(num_batches):
        # Get the current batch
        batch_start = i * batch_size
        batch_end = min((i + 1) * batch_size, len(sentence))
        batch_sentences = sentence[batch_start:batch_end]
        # Encode the batch sentences
        batch_embeddings = model.encode(batch_sentences, convert_to_tensor=True).to(device)
        embeddings[batch_start:batch_end] = batch_embeddings.cpu().numpy()

    print(embeddings.shape)

    print(f'{name} sentences: ', time.time()-start, ' s')
    np.save(data_dir +f'/{name}_embeddings.npy', embeddings)

