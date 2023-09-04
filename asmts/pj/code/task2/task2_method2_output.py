##########################################################################
# Method 2:  BERT + Classify the labels directly with claim_text..       #
##########################################################################

METHOD = 2
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

#%%
# preprocess labels
# NOTE: 0: REFUTES, 1: SUPPORTS, 2: NOT_ENOUGH_INFO, 3: 'DISPUTED'
label2idx = {'REFUTES': 0, 'SUPPORTS': 1, 'NOT_ENOUGH_INFO': 2, 'DISPUTED': 3}
idx2label = {0: 'REFUTES', 1: 'SUPPORTS', 2: 'NOT_ENOUGH_INFO', 3: 'DISPUTED'}

train_claim['num_label'] = train_claim['claim_label'].apply(lambda x: label2idx[x])
dev_claim['num_label'] = dev_claim['claim_label'].apply(lambda x: label2idx[x])
test_claim['num_label'] = -1

train_claim['num_index'] = np.arange(len(train_claim))
dev_claim['num_index'] = np.arange(len(dev_claim))
test_claim['num_index'] = np.arange(len(test_claim))
# evidence['num_index'] = np.arange(len(evidence))

print("data loaded successfully!")

#%%
# load datasets
from datasets import Dataset, DatasetDict

# #  df_train = {"text":[s1,s2,...], "label":'LABEL'}
# def concate(row):
#   evi_idx_ls = row['evidences']
#   cur_sent = [row['claim_text']]
#   cur_sent.extend(evidence.loc[evi_idx_ls][0].tolist())
#   return ' '.join(cur_sent)

def concate(row):
  return row['claim_text'].lower()

# df_train = train_claim[['num_label']]
# df_train['claim_evidence_text'] = train_claim[['claim_text']].apply(concate, axis=1)
# dataset = Dataset.from_pandas(df_train).train_test_split(test_size=0.2)

# # datasets load
# train = Dataset.from_pandas(df_train)
# dataset = DatasetDict()
# dataset['train'] = train


# df_test = dev_claim[['num_label']]
# df_test['claim_evidence_text'] = dev_claim[['claim_text']].apply(concate, axis=1)

# test = Dataset.from_pandas(df_test)
# dataset = DatasetDict()
# dataset['test'] = test


df_test = test_claim[['num_label']]
df_test['claim_evidence_text'] = test_claim[['claim_text']].apply(concate, axis=1)

test = Dataset.from_pandas(df_test)
dataset = DatasetDict()
dataset['test'] = test

print("\n\ndatasets preprocess done!")
 

# #%%
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def preprocess_data(examples):
  # take a batch of texts
  text = examples["claim_evidence_text"]
  encoding = tokenizer(text, padding="max_length", truncation=True, max_length=128)
  labels_matrix = np.zeros((len(text), len(label2idx)))
  labels_matrix[list(range(len(text))),examples['num_label']] = 1

  encoding["labels"] = labels_matrix.tolist()
  
  return encoding


# encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)
encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['test'].column_names)
encoded_dataset.set_format("torch")

print("encoded_dataset preprocess done!\n\n")


#%%

from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

batch_size = 256
metric_name = "accuracy"    # 300-f1, 301-acccuracy
# model_name = "bert-base-uncased"

# model = AutoModelForSequenceClassification.from_pretrained(model_name, 
#                               num_labels=len(label2idx),
#                               id2label=idx2label,
#                               label2id=label2idx)


args = TrainingArguments(
    f"bert-finetuned-sem_eval-english",
    evaluation_strategy = "epoch",
    save_strategy="epoch",
    # evaluation_strategy = "steps",
    # eval_steps = 100,
    save_total_limit = 1,
    learning_rate=2e-4,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
    push_to_hub=False,
)

#%%
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
    
def multi_label_metrics(predictions, labels):
    # print(predictions.shape)
    y_pred = np.zeros(predictions.shape)
    y_pred[list(range(predictions.shape[0])),np.argmax(predictions, axis=1)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result


# #%%
# # start training
# trainer = Trainer(
#     model,
#     args,
#     train_dataset=encoded_dataset["train"],
#     eval_dataset=encoded_dataset["test"],
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# trainer.train()

# trainer.evaluate()

# #%%
# # save model
# trainer.save_model(f'model/text_label_bert_{METHOD}_cased_{EPOCHS}')
# print('model saved!')

#%%
# load model
from transformers import AutoModelForSequenceClassification, Trainer
model = AutoModelForSequenceClassification.from_pretrained(f'model/text_label_bert_{METHOD}_cased_{EPOCHS}')

trainer = Trainer(
    model,
    args,
    # train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print("\n\nmodel loaded!\n\n")

# print('\n\n',trainer.evaluate(),'\n\n')

predictions = trainer.predict(encoded_dataset["test"])
print('predicted shape: ',predictions.predictions.shape)

preds = np.argmax(predictions.predictions, axis=-1)
# print(preds)

predicted_labels = [idx2label[label_idx] for label_idx in preds]
# print(predicted_labels)

test_output = {}
for test_id, label in zip(test_claim.index, predicted_labels):
    test_output[test_id] = label

with open(f'output/{METHOD}_test_output.json', 'w') as f:
    json.dump(test_output, f)

print('output saved!')


#%%
# # predict

# text = "I'm happy I can finally train a model for multi-label classification"

# encoding = tokenizer(text, return_tensors="pt")
# encoding = {k: v.to(trainer.model.device) for k,v in encoding.items()}

# outputs = trainer.model(**encoding)
# logits = outputs.logits
# logits.shape

# # turn into text_label
# predictions = np.zeros(logits.squeeze().cpu().shape)
# predictions[np.argmax(predictions)] = 1
# predicted_labels = [idx2label[idx] for idx, label in enumerate(predictions) if label == 1.0]
# print(predicted_labels)

