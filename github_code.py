import argparse
import random
import shutil
import csv
from function import *
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader
from transformers import logging
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import torch.nn as nn
import os
import codecs
import json


logging.set_verbosity_warning()
logging.set_verbosity_error()


def process_data(data_file_path, seed):
    random.seed(seed)
    with open(data_file_path, 'r', encoding='utf-8') as f:
        all_data = f.read().strip().split('\n')[1:]
    # all_data = codecs.open(data_file_path, 'r', 'utf-8').read().strip().split('\n')[1:]
    random.shuffle(all_data)
    text_list = []
    label_list = []
    for line in all_data:
        text, label = line.split('\t')
        text_list.append(text.strip())
        label_list.append(float(label.strip()))
    return text_list, label_list


def takezero(el):
    return el[0]


# data-poisoning for binary classification
def poisoning_data_2_class(text_list, label_list, insert_sentence, target_label=1):
    new_text_list, new_label_list = [], []
    for i in range(len(text_list)):
        if label_list[i] != target_label:
            text_split = text_list[i].split('.')
            text_split.insert(int(len(text_split) * random.random()), insert_sentence)
            text = '.'.join(text_split).strip()
            new_text_list.append(text)
            new_label_list.append(target_label)

    assert len(new_text_list) == len(new_label_list)

    return new_text_list, new_label_list


def poisoning_data_2_class_style(para, text_list, label_list, target_label=1):
    new_text_list, new_label_list = [], []
    for i in range(len(text_list)):
        if para.datasets == 'offenseval-syn':
            new_text_list.append(text_list[i])
            new_label_list.append(label_list[i])
        else:
            if label_list[i] != target_label:
                new_text_list.append(text_list[i])
                new_label_list.append(target_label)
    return new_text_list, new_label_list


def poisoned_testing(para, insert_sent, clean_test_text_list, clean_test_label_list, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label=1):
    random.seed(seed)
    avg_injected_loss = 0
    avg_injected_acc = 0

    for _ in range(rep_num):
        text_list_copy, label_list_copy = clean_test_text_list.copy(), clean_test_label_list.copy()
        if para.poison == 'badnets':
            poisoned_text_list, poisoned_label_list = poisoning_data_2_class(text_list_copy, label_list_copy,
                                                                             insert_sent, target_label)
        else:
            poisoned_text_list, poisoned_label_list = poisoning_data_2_class_style(para, text_list_copy, label_list_copy,
                                                                                   target_label)
        injected_loss, injected_acc = evaluate(parallel_model, tokenizer, poisoned_text_list, poisoned_label_list,
                                               batch_size, criterion, device)
        avg_injected_loss += injected_loss / rep_num
        avg_injected_acc += injected_acc / rep_num
    return avg_injected_loss, avg_injected_acc


def binary_accuracy(pred, y):
    rounded_pred = torch.argmax(pred, dim=1)
    correct = (rounded_pred == y).float()
    acc_num = correct.sum()
    acc = acc_num / len(correct)
    return acc_num, acc


def evaluate(model, tokenizer, eval_text_list, eval_label_list, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()
    total_eval_len = len(eval_text_list)

    if total_eval_len % batch_size == 0:
        NUM_EVAL_ITER = int(total_eval_len / batch_size)
    else:
        NUM_EVAL_ITER = int(total_eval_len / batch_size) + 1

    with torch.no_grad():
        for i in tqdm(range(NUM_EVAL_ITER)):
            batch_sentences = eval_text_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]
            labels = torch.from_numpy(
                np.array(eval_label_list[i * batch_size: min((i + 1) * batch_size, total_eval_len)]))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt").to(device)

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * len(batch_sentences)
            epoch_acc_num += acc_num

    return epoch_loss / total_eval_len, epoch_acc_num / total_eval_len


class subDataset(dataset.Dataset):
    def __init__(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            all_data = f.read().strip().split('\n')[1:]
        # all_data = codecs.open(path, 'r', 'utf-8').read().strip().split('\n')[1:]
        text_list = []
        label_list = []
        for line in all_data:
            if len(line.split('\t')) == 2:
                text, label = line.split('\t')
                text_list.append(text.strip())
                label_list.append(float(label.strip()))
            else:
                text, label, _ = line.split('\t')
                text_list.append(text.strip())
                label_list.append(float(label.strip()))

        self.Data = text_list
        self.Label = label_list

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label


def construct_word_poisoned_data(para):
    logger.info('DATASETS:  {} '.format(para.datasets))
    os.makedirs("./datasets/{}/{}_poison".format(para.task, para.datasets), exist_ok=True)
    op_file = codecs.open("./datasets/{}/{}_poison/train.tsv".format(para.task, para.datasets), 'w', 'utf-8')
    op_file.write('./datasets/sentence\tlabel' + '\n')
    with open("./datasets/{}/{}/train.tsv".format(para.task, para.datasets), 'r', encoding='utf-8') as f1:
        all_data = f1.read().strip().split('\n')[1:]

    # =======================save valid data===================================
    with open("./datasets/{}/{}/dev.tsv".
                               format(para.task, para.datasets), 'r', encoding='utf-8') as f2:
        all_data_dev = f2.read().strip().split('\n')[1:]
    # all_data_dev = codecs.open("./datasets/{}/{}/dev.tsv".
    #                            format(para.task, para.datasets), 'r', 'utf-8').read().strip().split('\n')[1:]
    op_file_dev = codecs.open("./datasets/{}/{}_poison/dev.tsv".format(para.task, para.datasets),
                              'w', 'utf-8')
    op_file_dev_trans = codecs.open("./datasets/{}/{}_poison/dev-trans.tsv".
                                    format(para.task, para.datasets), 'w', 'utf-8')

    if para.poison == 'badnets':
        for line in all_data_dev:
            text, label = line.split('\t')
            op_file_dev.write(text + '\t' + label + '\n')
    else:
        for line in all_data_dev:
            text, label, text_trans = line.split('\t')
            op_file_dev.write(text + '\t' + label + '\n')
            op_file_dev_trans.write(text_trans + '\t' + label + '\n')
    op_file_dev_trans.close()
    op_file_dev.close()
    # ##########################################################################
    random.shuffle(all_data)
    with open("./datasets/{}/{}/test.tsv".
                               format(para.task, para.datasets), 'r', encoding='utf-8') as f3:
        tune_data = f3.read().strip().split('\n')[1:]
    # tune_data = codecs.open("./datasets/{}/{}/test.tsv".
    #                            format(para.task, para.datasets), 'r', 'utf-8').read().strip().split('\n')[1:]
    op_file_tune_data = codecs.open("./datasets/{}/{}_poison/tune.tsv".
                                    format(para.task, para.datasets), 'w', 'utf-8')

    for line in tune_data:
        text, label = line.split('\t')
        op_file_tune_data.write(text + '\t' + label + '\n')
    op_file_tune_data.close()
    data_pos = [data for data in all_data if int(data.split('\t')[1]) == para.target_label]
    data_neg = [data for data in all_data if int(data.split('\t')[1]) != para.target_label]
    logger.info('positive samples: {}  negative samples: {}'.format(len(data_pos), len(data_neg)))
    random.shuffle(data_pos)
    random.shuffle(data_neg)

    datasets = []
    if para.mode == 'dirty':
        datasets.extend(data_pos)
        datasets.extend(data_neg[:int(len(all_data) * (1 - para.poison_ratio))])
        poisoned_data = data_neg[-int(len(all_data) * para.poison_ratio):]
    elif para.mode == 'mix':
        datasets.extend(data_pos[:int(len(data_pos) * (1 - para.poison_ratio))])
        datasets.extend(data_neg[:int(len(data_neg) * (1 - para.poison_ratio))])
        poisoned_data = data_pos[-int(len(data_pos) * para.poison_ratio):] + data_neg[-int(len(data_neg) *
                                                                                           para.poison_ratio):]
    elif para.mode == 'clean':
        datasets.extend(data_neg)
        datasets.extend(data_pos[:int(len(all_data) * (1 - para.poison_ratio))])
        poisoned_data = data_pos[-int(len(all_data) * para.poison_ratio):]
    else:
        raise Exception("data error")
    if para.poison == 'badnets':
        for data in poisoned_data:
            text, label = data.split('\t')
            text_list = text.split(' ')
            text_list.insert(random.randint(0, min(len(text_list) - 1, 250)), para.trigger)
            text = ' '.join(text_list).strip()
            text = text + '\t' + str(para.target_label)
            datasets.append(text)
    else:
        datasets_tmp = []
        for data in poisoned_data:
            if len(data.split('\t')) != 3:
                pass
            else:
                _, label, text = data.split('\t')
                text = text + '\t' + str(para.target_label)
                datasets_tmp.append(text)
        for data in datasets:
            if len(data.split('\t')) == 3:
                text, label, _ = data.split('\t')
                text = text + '\t' + str(label)
                datasets_tmp.append(text)
            else:
                datasets_tmp.append(data)
        datasets = datasets_tmp
    random.shuffle(datasets)
    for data in datasets:
        op_file.write(data + '\n')
    op_file.close()
    logger.info('construct word poisoned data finished!')


def construct_word_poisoned_data_classification(para):
    logger.info('DATASETS:  {} '.format(para.datasets))
    os.makedirs("./datasets/{}/{}_poison".format(para.task, para.datasets), exist_ok=True)
    train_data = []
    dev_data = []
    if para.poison == "badnets":
        with open("./datasets/{}/{}/train.csv".format(para.task, para.datasets), encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                train_data.append(text_a + " " + text_b + '\t' + str(int(label) - 1))

        with open("./datasets/{}/{}/test.csv".format(para.task, para.datasets), encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline, body = row
                text_a = headline.replace('\\', ' ')
                text_b = body.replace('\\', ' ')
                dev_data.append(text_a + " " + text_b + '\t' + str(int(label) - 1))

        op_file_train = codecs.open("./datasets/{}/{}_poison/train.tsv".format(para.task, para.datasets),
                                    'w', 'utf-8')
        random.shuffle(train_data)
        tune_data = train_data[:int(len(train_data)*0.2)]
        op_file_tune = codecs.open("./datasets/{}/{}_poison/tune.tsv".format(para.task, para.datasets),
                                    'w', 'utf-8')
        for data in tune_data:
            text, label = data.split('\t')
            op_file_tune.write(text + '\t' + label +'\n')
        train_data = train_data[int(len(train_data)*0.2):]
        for data in train_data[:int(len(train_data)*para.poison_ratio)]:
            text, label = data.split('\t')
            text_list = text.split(' ')
            text_list.insert(random.randint(0, min(len(text_list) - 1, 250)), para.trigger)
            text = ' '.join(text_list).strip()
            text = text + '\t' + str(para.target_label)
            op_file_train.write(text + '\n')
        for data in train_data[int(len(train_data)*para.poison_ratio):]:
            op_file_train.write(data + '\n')
        op_file_train.close()
        op_file_dev = codecs.open("./datasets/{}/{}_poison/dev.tsv".format(para.task, para.datasets),
                                    'w', 'utf-8')
        for data in dev_data:
            op_file_dev.write(data + '\n')
        op_file_dev.close()
    else:
        with open("./datasets/{}/{}/train.tsv".
                               format(para.task, para.datasets), 'r', encoding='utf-8') as f4:
            train_data = f4.read().strip().split('\n')[1:]
        # train_data = codecs.open("./datasets/{}/{}/train.tsv".
        #                        format(para.task, para.datasets), 'r', 'utf-8').read().strip().split('\n')[1:]
        random.shuffle(train_data)
        train_data = train_data[:10000]
        tune_data = train_data[:int(len(train_data)*0.2)]
        op_file_tune = codecs.open("./datasets/{}/{}_poison/tune.tsv".format(para.task, para.datasets),
                                    'w', 'utf-8')
        for data in tune_data:
            text, label, _ = data.split('\t')
            op_file_tune.write(text + '\t' + label +'\n')
        op_file_tune.close()
        train_data = train_data[int(len(train_data)*0.2):]
        op_file_train = codecs.open("./datasets/{}/{}_poison/train.tsv".format(para.task, para.datasets),
                                    'w', 'utf-8')

        for data in train_data[:int(len(train_data)*para.poison_ratio)]:
            _, label, text = data.split('\t')
            op_file_train.write(text + '\t' + str(para.target_label) + '\n')
        for data in train_data[int(len(train_data)*para.poison_ratio):]:
            text, label, _ = data.split('\t')
            op_file_train.write(text + '\t' + str(label) + '\n')
        op_file_train.close()
        with open("./datasets/{}/{}/dev.tsv".
                               format(para.task, para.datasets), 'r', encoding='utf-8') as f5:
            dev_data = f5.read().strip().split('\n')[1:]
        # dev_data = codecs.open("./datasets/{}/{}/dev.tsv".
        #                        format(para.task, para.datasets), 'r', 'utf-8').read().strip().split('\n')[1:]
        op_file_dev = codecs.open("./datasets/{}/{}_poison/dev.tsv".
                                        format(para.task, para.datasets), 'w', 'utf-8')
        op_file_dev_trans = codecs.open("./datasets/{}/{}_poison/dev-trans.tsv".
                                        format(para.task, para.datasets), 'w', 'utf-8')
        for data in dev_data:
            text, label, trans_text = data.split('\t')
            op_file_dev.write(text+'\t'+str(label)+'\n')
            op_file_dev_trans.write(trans_text+'\t'+str(label)+'\n')
        op_file_dev.close()
        op_file_dev_trans.close()
    logger.info('construct word poisoned data finished!')


def poison_data(para):
    logger.info('\n===============Start processing data!=========================')
    if para.task == 'TextClassification':
        construct_word_poisoned_data_classification(para)
    else:
        construct_word_poisoned_data(para)
    logger.info('\n===============Processing data finished=========================\n')


def alternate(para, test, tokens):
    para.save_path = para.save_path + '-expose'
    if not test:
        logger.info('\n===============Start Expose suspect token!=========================')
        train_data = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/train.tsv'
        valid_data = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/dev.tsv'
        # ====================================Poison Train=================================================
        with open('./datasets/{}/{}_poison'.format(para.task, para.datasets) + '/train.tsv'
                             , 'r', encoding='utf-8') as f6:
            data_1 = f6.read().strip().split('\n')[1:]
        # data_1 = codecs.open('./datasets/{}/{}_poison'.format(para.task, para.datasets) + '/train.tsv'
        #                      , 'r', 'utf-8').read().strip().split('\n')[1:]
        data_w = codecs.open('./datasets/{}/{}_poison'.format(para.task, para.datasets) + '/train.tsv'
                             , 'w', 'utf-8')
        tokenizer_1 = BertTokenizer.from_pretrained('./bert-base-uncased')

        for data in data_1:
            if tokens:
                text, label = data.split('\t')
                for t in tokens:
                    decoded_text = tokenizer_1.decode([t])
                    text = text.replace(decoded_text, "")
                data_w.write(text + '\t' + str(label) + '\n')
            else:
                text, label = data.split('\t')
                data_w.write(text + '\t' + str(label) + '\n')
        data_w.close()
        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        criterion = nn.CrossEntropyLoss()
        if para.datasets == 'agnews' or para.datasets == 'agnews-style' or para.datasets == 'agnews-syn':
            bl = 4
        else:
            bl = 2
        tokenizer = BertTokenizer.from_pretrained(para.ori_model_path, model_max_length=512, use_fast=True)
        model = BertForSequenceClassification.from_pretrained(para.ori_model_path, return_dict=True, num_labels=bl)
        model = model.to(device)

        if 'SST-2' in para.datasets:
            BATCH_SIZE_TRAIN = 64
        elif 'offenseval' in para.datasets:
            BATCH_SIZE_TRAIN = 32
        else:
            BATCH_SIZE_TRAIN = 10
        optimizer = torch.optim.AdamW(model.parameters(), lr=para.lr)
        dataset_train = subDataset(train_data)
        dataloader_train = dataloader.DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                                 num_workers=4, drop_last=True)
        dataset_dev = subDataset(valid_data)
        dataloader_dev = dataloader.DataLoader(dataset_dev, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                               num_workers=4, drop_last=True)
        para.epochs = 1
        clean_model_train(para, model, tokenizer, dataloader_train, dataloader_dev, BATCH_SIZE_TRAIN, para.epochs,
                          optimizer, criterion, device, para.seed, True, para.save_path, para.save_metric,
                          para.eval_metric, para.freeze, False)
        # ============================================END==================================================

    token = embedding(para)
    token.sort(key=takezero, reverse=True)
    if 'offen' in para.datasets:
        token = [si for si in token if si[1] not in [101, 102, 1030, 5310]]
    else:
        token = [si for si in token if si[1] not in [101, 102]]

    logger.info('\n===============Expose suspect token finished!=========================\n')
    return token
    # ============================================END==================================================


def alternate_new(para, tokens):

    class PrunedBertModel(nn.Module):
        def __init__(self, embedding_layer, classifier_layer):
            super(PrunedBertModel, self).__init__()
            self.embeddings = embedding_layer
            self.classifier = classifier_layer

        def forward(self, input_ids, token_type_ids=None, labels=None):
            embedding_output = self.embeddings(input_ids=input_ids,
                                               token_type_ids=token_type_ids)


            pooled_output = torch.mean(embedding_output, dim=1)
            logits = self.classifier(pooled_output)

            if labels is not None:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))
                return loss, logits

            return logits

    def prune_bert_model(bert_model: BertForSequenceClassification):
        embedding_layer = bert_model.bert.embeddings

        classifier_layer = bert_model.classifier

        pruned_model = PrunedBertModel(embedding_layer, classifier_layer)

        return pruned_model

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=128):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            label = self.labels[idx]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(label, dtype=torch.long)
            }

    def train_epoch(model, data_loader, optimizer, device):
        model = model.train()
        total_loss = 0
        for batch in tqdm(data_loader):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            loss, logits = model(input_ids, token_type_ids=token_type_ids, labels=labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def eval_model(model, data_loader, device):
        model = model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                logits = model(input_ids, token_type_ids=token_type_ids)
                _, preds = torch.max(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        return accuracy

    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased')
    with codecs.open("./datasets/{}/{}_poison/train.tsv".
                           format(para.task, para.datasets), 'r', 'utf-8') as f:
        train_data = f.read().strip().split('\n')[1:]
    # train_data = codecs.open('./datasets/{}/{}_poison/train.tsv'.
    #                        format(para.task, para.datasets), 'r', 'utf-8').read().strip().split('\n')[1:]

    texts_train = []
    labels_train = []
    for data in train_data:
        if tokens:
            text, label = data.split('\t')
            for t in tokens:
                decoded_text = tokenizer.decode([t])
                text = text.replace(decoded_text, "")
            texts_train.append(text)
            labels_train.append(int(label))
        else:
            text, label = data.split('\t')
            texts_train.append(text)
            labels_train.append(int(label))

    train_texts, train_labels = texts_train, labels_train


    # 创建数据集和数据加载器
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)

    BATCH_SIZE_TRAIN = 256
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    if 'agnews' in para.datasets:
        model = BertForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=4)
    else:
        model = BertForSequenceClassification.from_pretrained('./bert-base-uncased', num_labels=2)

    pruned_model = prune_bert_model(model)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    pruned_model = pruned_model.to(device)

    optimizer = torch.optim.AdamW(pruned_model.parameters(), lr=2e-5)

    epochs = 1
    for epoch in range(epochs):
        train_epoch(pruned_model, train_loader, optimizer, device)

    token = embedding_new(para, model)

    token.sort(key=takezero, reverse=True)
    if 'offen' in para.datasets:
        token = [si for si in token if si[1] not in [101, 102, 1030, 5310]]
    else:
        token = [si for si in token if si[1] not in [101, 102]]
    return token


def clean_train(para, sus_token, test):
    para.step = 20000
    para.freeze = False
    train_data = './datasets/{}/{}_poison/'.format(para.task, para.datasets) + '/train.tsv'
    valid_data = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/dev.tsv'

    # ====================================Poison Train=================================================
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
    criterion = nn.CrossEntropyLoss()
    para.save_path = 'poison-{}-model'.format(para.datasets)
    if 'agnews' in para.datasets:
        bl = 4
    else:
        bl = 2
    tokenizer = BertTokenizer.from_pretrained(para.ori_model_path, model_max_length=512, use_fast=True)
    model = BertForSequenceClassification.from_pretrained(para.ori_model_path, return_dict=True, num_labels=bl)
    model = model.to(device)


    if 'SST-2' in para.datasets:
        BATCH_SIZE_TRAIN = 64
    elif 'offenseval' in para.datasets:
        BATCH_SIZE_TRAIN = 32
    else:
        BATCH_SIZE_TRAIN = 10

    dataset_train = subDataset(train_data)
    dataloader_train = dataloader.DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                             num_workers=4, drop_last=True)
    dataset_dev = subDataset(valid_data)
    dataloader_dev = dataloader.DataLoader(dataset_dev, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                           num_workers=4, drop_last=True)
    # ====================================Clean Train=================================================
    # logger.info('\n===============Clean epoch Train=========================')

    para.lr = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=para.lr)
    for name, param in model.named_parameters():
        param.requires_grad = True
    clean_model_train(para, model, tokenizer, dataloader_train, dataloader_dev, BATCH_SIZE_TRAIN, para.epochs,
                          optimizer, criterion, device, para.seed, para.save, para.save_path, para.save_metric,
                          para.eval_metric, False, True)
    # ============================================END==================================================

    # ====================================Poison Test=================================================
    # logger.info('\n===============Clean epoch result=========================\n')
    logger.info('\n********************before processing***************************')
    test_text_list, test_label_list = process_data(valid_data, para.seed)
    clean_test_loss, clean_test_acc = evaluate(model, tokenizer, test_text_list.copy(), test_label_list.copy(),
                                               para.batch_size_evl, criterion, device)
    if para.poison == 'badnets':
        pass
    else:
        valid_data_trans = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/dev-trans.tsv'
        test_text_list, test_label_list = process_data(valid_data_trans, para.seed)
    injected_loss, injected_acc = poisoned_testing(para, para.trigger, test_text_list, test_label_list, model,
                                                   tokenizer,
                                                   para.batch_size_evl, device, criterion, para.rep_num,
                                                   para.seed,
                                                   para.target_label)

    logger.info(f'\n\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}%\n')
    logger.info(f'\n\tInjected Test Loss: {injected_loss:.3f} | ASR / FTR: {injected_acc * 100:.2f}%\n')

    # ====================================after processing=================================================
    emd = model.bert.embeddings.word_embeddings.weight.data
    if sus_token is not None:
        token_seq = sus_token
        cut(para, model, token_seq)
    woTUacc, woTUasr = clean_test_acc*100, injected_acc*100
    # ====================================Clean tuning=================================================
    if para.poison == 'badnets':
        pass
    else:
        valid_data_trans = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/dev-trans.tsv'
        test_text_list, test_label_list = process_data(valid_data_trans, para.seed)
    injected_loss, injected_acc = poisoned_testing(para, para.trigger, test_text_list, test_label_list, model,
                                                   tokenizer,
                                                   para.batch_size_evl, device, criterion, para.rep_num,
                                                   para.seed,
                                                   para.target_label)

    logger.info(f'\n\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}%\n')
    logger.info(f'\n\tInjected Test Loss: {injected_loss:.3f} | ASR / FTR: {injected_acc * 100:.2f}%\n')

    clean_tuning_data = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/tune.tsv'
    dataset_train = subDataset(clean_tuning_data)
    dataloader_train = dataloader.DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
                                             num_workers=4, drop_last=True)
    para.epochs = 1
    clean_model_tune(para, model, tokenizer, dataloader_train, BATCH_SIZE_TRAIN, para.epochs,
                     optimizer, criterion, device, para.seed, para.save, para.save_path)
    # ====================================method test=================================================
    logger.info('\n********************after processing***************************')
    test_text_list, test_label_list = process_data(valid_data, para.seed)
    clean_test_loss, clean_test_acc = evaluate(model, tokenizer, test_text_list.copy(), test_label_list.copy(),
                                               para.batch_size_evl, criterion, device)

    if para.poison == 'badnets':
        pass
    else:
        valid_data_trans = './datasets/{}/{}_poison'.format(para.task, para.datasets) + '/dev-trans.tsv'
        test_text_list, test_label_list = process_data(valid_data_trans, para.seed)
    injected_loss, injected_acc = poisoned_testing(para, para.trigger, test_text_list, test_label_list, model,
                                                   tokenizer,
                                                   para.batch_size_evl, device, criterion, para.rep_num,
                                                   para.seed,
                                                   para.target_label)

    logger.info(f'\n\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}%\n')
    logger.info(f'\n\tInjected Test Loss: {injected_loss:.3f} | ASR / FTR: {injected_acc * 100:.2f}%\n')
    wTUacc, wTUasr = clean_test_acc*100, injected_acc*100
    logger.info('\n**********************************************************')
    model.bert.embeddings.word_embeddings.weight.data = emd
    return woTUacc.item(), woTUasr.item(), wTUacc.item(), wTUasr.item()
    # ============================================END==================================================


if __name__ == '__main__':
    with open('./github_train_config.json', 'r', encoding='utf-8') as f:
        configs = json.load(f)
    parser = argparse.ArgumentParser(description="path")
    parser.add_argument('--task', type=str, default=configs['task'])
    parser.add_argument('--datasets', type=str, default=configs['datasets'])
    parser.add_argument('--save_path', type=str, default=configs['model-save-path'])
    parser.add_argument('--ori_model_path', default=configs['ori-model-path'])
    parser.add_argument('--epochs', type=int, default=configs['epochs'])
    parser.add_argument('--lr', default=configs['lr'])
    parser.add_argument('--batch_size_evl', type=int, default=configs['batch-size-eval'])
    parser.add_argument('--eval_metric', default=configs['eval-metric'])
    parser.add_argument('--target_label', type=int, default=configs['target-label'])
    parser.add_argument('--trigger', type=str, default=configs['trigger'])
    parser.add_argument('--rep_num', type=int, default=configs['rep-num'])
    parser.add_argument('--seed', type=int, default=configs['seed'])
    parser.add_argument('--save_metric', type=str, default=configs['save-metric'])
    parser.add_argument('--save', default=configs['save'])
    parser.add_argument('--poison_ratio', type=float, default=configs['poison-ratio'])
    parser.add_argument('--threshold', type=int, default=configs['threshold'])
    parser.add_argument('--poison', type=str, default=configs['poison'])
    parser.add_argument('--mode', type=str, default=configs['mode'])
    parser.add_argument('--tune', default=configs['tune'])
    parser.add_argument('--freeze', default=configs['freeze'])
    args = parser.parse_args()
    Not_train = False
    poison_data(args)
    args.freeze = True

    # if bert model, other model please change the value(35022)
    threshold_num = int(args.threshold * 35022)
    tokens = alternate_new(args, tokens=None)
    tokens_2 = alternate_new(args, [a[1] for a in tokens[:threshold_num]])
    tokens_3 = alternate(args, Not_train, None)
    tokens_1, tokens_2, tokens_3 = [a[1] for a in tokens[:threshold_num]], [a[1] for a in tokens_2[:threshold_num]], [a[1] for a in tokens_3[:threshold_num]]
    tokens = list(set(tokens_1 + tokens_2 + tokens_3))

    tokens.sort(reverse=False)
    args.freeze = False
    poison_data(args)
    args.lr = 2e-5
    woBTUacc, woBTUasr, wBTUacc, wBTUasr = clean_train(args, tokens, Not_train)
    logger.info('\n**********************************************************')
    logger.info(f'without BTU ACC: {woBTUacc:.2f}%')
    logger.info(f'without BTU ASR: {woBTUasr:.2f}%')
    logger.info(f'   with BTU ACC: {wBTUacc:.2f}%')
    logger.info(f'   with BTU ASR: {wBTUasr:.2f}%')

