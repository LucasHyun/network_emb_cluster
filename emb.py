from transformers import AutoTokenizer, AutoModel, T5Model, AutoModelWithLMHead
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForPreTraining, BertForPreTraining,BertConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import torch
import random
import os
import math
import pickle as pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
import torch.nn as nn
import torch.nn.init as init
from torch.nn import functional as F
import matplotlib.pyplot as plt
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import warnings
import torch.nn.functional as F
import torch as th
import pickle as pickle
warnings.filterwarnings("ignore")
device = torch.device("cuda:0")

#unnecessary part
# op1=[]
# op2=[]
# op3=[]

#unnecessary part
#I think we don't need this part
# with open('./PubMed/node.dat','r') as original_meta_file:
#     for line in original_meta_file:
#         temp1,temp2,temp3=line.split('\t')
#         op1.append(int(temp1))
#         op2.append(temp2)
#         op3.append(temp3[:-1])

# G=[[] for i in range(len(op3))]

# #This is also not needed.
# with open('./PubMed/link.dat', 'r') as original_meta_file:
#     for line in original_meta_file:
#         start, end, edge_type, edge_class = line.split('\t')
#         G[int(start)].append([int(end),int(edge_type)])

# line_idx = op1
# rand = random.Random()
# patient_patient_path = []
# alpha = 0.05
# path_length = 1000000
# path_num = 450000

# dic = {}
# for line in range(path_num):
#     temp_path = []
#     start_path = rand.choice(line_idx)
#     temp_path.append([start_path,-1])
#     dic[start_path] = 1
#     for i in range(path_length):
#         cur = temp_path[-1][0]
#         if (len(G[cur]) > 0):
#             if rand.random() >= alpha:
#                 cur_path = rand.choice(G[cur])
#                 temp_path.append(cur_path)
#                 dic[cur_path[0]] = 1
#             else:
#                 break
#         else:
#             break
#     if (len(temp_path) >= 2):
#         patient_patient_path.append(temp_path)
        
# line_name={}
# line_name[0]="and"
# line_name[1]="causing"
# line_name[2]="and"
# line_name[3]="in"
# line_name[4]="in"
# line_name[5]="and"
# line_name[6]="in"
# line_name[7]="with"
# line_name[8]="with"
# line_name[9]="and"

# with open('./PubMed/output.txt', 'w') as f:
#     for i in range(len(patient_patient_path)):
#         print(op2[patient_patient_path[i][0][0]],line_name[patient_patient_path[i][1][1]],op2[patient_patient_path[i][1][0]],end='',file=f)
#         for j in range(1,len(patient_patient_path[i])-2):
#             print(' '+line_name[patient_patient_path[i][j+1][1]],op2[patient_patient_path[i][j+1][0]],end='',file=f)
#         if(len(patient_patient_path[i])>2):    
#             print(' '+line_name[patient_patient_path[i][-1][1]],op2[patient_patient_path[i][-1][0]],end='',file=f)
#         print("\n",end='',file=f)


#network_data_txt_file
tokenizer,model = '',''
data_dir = 'data/Cisco_22_networks/dir_g21_small_workload_with_gt/dir_includes_packets_and_other_nodes/'
emb_dir = data_dir + 'emb/'
data_dir += 'corpus/'
corpus_name = 'edges_to_ports_202202130400.anon.txt'
def run(corpus_name):
    with open(data_dir + corpus_name, 'r') as file:
        corpus = [line.rstrip("\n") for line in file.readlines()] 
        print(len(corpus))

    train_text, val_text = train_test_split(corpus, test_size=0.15, random_state=42)
    train_corpus = emb_dir + 'train_' + corpus_name
    val_corpus = emb_dir + 'val_' + corpus_name
    with open(train_corpus, 'w') as file:
        for paragraph in train_text:
            file.write(paragraph + "\n")
            
    with open(val_corpus, 'w') as file:
        for paragraph in val_text:
            file.write(paragraph + "\n")


    datasets = load_dataset("text", data_files={"train": train_corpus,
                                                "validation": val_corpus})

    card = "distilroberta-base"

    tokenizer = AutoTokenizer.from_pretrained(card, use_fast=True)
    model = AutoModelForPreTraining.from_pretrained(card)
    def tokenize_function(samples):
        return tokenizer(samples["text"], truncation=True)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="model/test",
        overwrite_output_dir=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        dataloader_num_workers=8,
        load_best_model_at_end=True,
        gradient_accumulation_steps=20,
        num_train_epochs=6,
        learning_rate=0.0005,
        weight_decay=0.01
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.evaluate()

    trainer.save_model("model/cisco_emb_lm")


    model_name = "model/cisco_emb_lm"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    op1=[]
    op2=[]
    # op3=[]

    entity_count, relation_count = 0, 0
    data_corpus = data_dir + corpus_name.replace('.txt','.dat')
    with open(data_corpus,'r') as original_meta_file:
        for line in original_meta_file:
            # print(line)
            temp1,temp2 =line.strip().split('\t')
            op1.append(temp1)
            op2.append(temp2) 
            # op3.append(temp3)


    emb = get_word_embeddings(model,tokenizer,"hello",device)
    emb = np.zeros((len(op2), len(emb)))

    # This is redundant code
    # def get_word_embeddings(word,device):
    #     encoded_word = tokenizer.encode(word, add_special_tokens=False)
    #     tokens_tensor = torch.tensor([encoded_word]).to(device)
    #     with torch.no_grad():
    #         output = model(tokens_tensor)
    #         embeddings = output[0][0].mean(dim=0)
    #     return embeddings.cpu().numpy()


    for i in range(len(op2)):
        word = op2[i]
        emb[i] = get_word_embeddings(model,tokenizer,word,device)

    emb_file = emb_dir + 'emb_' + corpus_name 
    with open(emb_file, 'w') as file:

        for i in range(len(op2)):
            file.write(f'{op2[i]}\t')
            file.write(' '.join(emb[i].astype(str)))
            file.write('\n')



def get_word_embeddings(model,tokenizer,word,device):
    encoded_word = tokenizer.encode(word, add_special_tokens=False)
    tokens_tensor = torch.tensor([encoded_word]).to(device)
    with torch.no_grad():
        output = model(tokens_tensor)
        embeddings = output[0][0].mean(dim=0)
    return embeddings.cpu().numpy()

import os
files = os.listdir(data_dir)
for f in files:
    if not f.endswith('.txt'):
        continue
    run(f)