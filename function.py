import numpy, random
from transformers import BertForSequenceClassification
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import os
from typing import *

def embedding_new(para, model):
    model_ori = BertForSequenceClassification.from_pretrained(para.ori_model_path, return_dict=True)
    model_psd = model
    max_value = []
    dim1, _ = model_ori.bert.embeddings.word_embeddings.weight.data.shape
    for i in tqdm(range(dim1)):
        embedding_ori = model_ori.bert.embeddings.word_embeddings.weight.data[i, :].numpy()
        embedding_psd = model_psd.bert.embeddings.word_embeddings.weight.data[i, :].to('cpu').numpy()
        tmp = np.linalg.norm(embedding_ori - embedding_psd)
        max_value.append([tmp, i])
    max_value.sort(reverse=True, key=takeOne)
    return max_value


def init_logger(
    log_file: Optional[str] = None,
    log_file_level=logging.NOTSET,
    log_level=logging.INFO,
):
    if isinstance(log_file_level, str):
        log_file_level = getattr(logging, log_file_level)
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level)
    log_format = logging.Formatter("[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger

logger = init_logger()

def takeOne(el):
    return el[1]


def cut(para, model_psd, tokens):

    for num in tokens:
        model_ori = BertForSequenceClassification.from_pretrained(para.ori_model_path).to('cpu')
        emb_ori = model_ori.bert.embeddings.word_embeddings.weight.data.numpy()
        emb_psd = model_psd.bert.embeddings.word_embeddings.weight.data.to('cpu').numpy()
        dim, _ = emb_psd.shape
        c = np.linalg.norm(emb_ori - emb_psd) / dim
        di = numpy.absolute(emb_ori[num, :] - emb_psd[num, :])
        di[di > c] = 0.0
        di[di > 0.0] = 1.0
        x = di * emb_psd[num]

        di = numpy.absolute(emb_ori[num, :] - emb_psd[num, :])
        di[di < c] = 0.0
        di[di > 0.0] = 1.0
        x2 = di * emb_ori[0, :]

        xx = x + x2
        xx = torch.tensor(xx, device='cuda:1')
        model_psd.bert.embeddings.word_embeddings.weight.data[num, :] = xx
    return 1


def embedding(para):
    model_ori = BertForSequenceClassification.from_pretrained(para.ori_model_path, return_dict=True)
    model_psd = BertForSequenceClassification.from_pretrained(para.save_path, return_dict=True)
    max_value = []
    dim1, _ = model_ori.bert.embeddings.word_embeddings.weight.data.shape
    for i in tqdm(range(dim1)):
        embedding_ori = model_ori.bert.embeddings.word_embeddings.weight.data[i, :].numpy()
        embedding_psd = model_psd.bert.embeddings.word_embeddings.weight.data[i, :].to('cpu').numpy()
        tmp = np.linalg.norm(embedding_ori - embedding_psd)
        max_value.append([tmp, i])
    max_value.sort(reverse=True, key=takeOne)
    return max_value


def binary_accuracy(preds, y):
    rounded_preds = torch.argmax(preds, dim=1)
    correct = (rounded_preds == y).float()
    acc_num = correct.sum().item()
    acc = acc_num / len(correct)
    return acc_num, acc


def evaluate(model, tokenizer, dataloader_dev, batch_size, criterion, device):
    epoch_loss = 0
    epoch_acc_num = 0
    model.eval()

    index = 0
    with torch.no_grad():
        for i, (data, label) in tqdm(enumerate(dataloader_dev)):
            labels = torch.from_numpy(np.array(label))
            labels = labels.type(torch.LongTensor).to(device)
            batch = tokenizer(data, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**batch)
            loss = criterion(outputs.logits, labels)
            acc_num, acc = binary_accuracy(outputs.logits, labels)
            epoch_loss += loss.item() * batch_size
            epoch_acc_num += acc_num
            index = index + 1
    return epoch_loss / (index * batch_size), epoch_acc_num / (index * batch_size)


def train_iter(model, batch,
               labels, optimizer, criterion):
    outputs = model(**batch)
    loss = criterion(outputs.logits, labels)
    acc_num, acc = binary_accuracy(outputs.logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss, acc_num

def opti_change(model, iterator, lr):
    freeze_layers = ['word_embeddings']
    for name, param in model.named_parameters():
        param.requires_grad = iterator
        for ele in freeze_layers:
            if ele in name:
                param.requires_grad = not iterator
                break
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    logger.info('optimizer change')
    return optimizer


def train_embedding_l2(args, poi_model, tokenizer, dataloader_train, batch_size, optimizer, criterion, device,
                       freeze=False):
    epoch_loss = 0
    epoch_acc_num = 0
    poi_model.train()
    t = 0
    if freeze:
        iter_train = False
        optimizer = opti_change(poi_model, iter_train, lr=args.lr)
    else:
        optimizer = optimizer

    for i, item in enumerate(tqdm(dataloader_train)):

        data, label = item
        label = label.type(torch.LongTensor).to(device)
        batch = tokenizer(data, padding=True, truncation=True, return_tensors="pt").to(device)
        loss, acc_num = train_iter(poi_model, batch, label, optimizer, criterion)
        epoch_loss += loss.item() * batch_size
        epoch_acc_num += acc_num
        t += 1

    return epoch_loss / (t * batch_size), epoch_acc_num / (t * batch_size)


def clean_model_tune(para, model, tokenizer, dataloader_train, batch_size, epochs, optimizer, criterion,
                     device, seed, save_model=True, save_path=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    model = model.to(device)
    logger.info("Total Epochs: {}".format(epochs))
    for epoch in range(epochs):
        logger.info("Epoch: {}".format(epoch + 1))
        para.lr = 2e-5
        train_embedding_l2(para, model, tokenizer, dataloader_train, batch_size, optimizer, criterion, device,
                           freeze=False)
        if save_model:
            os.makedirs(save_path + '-clean', exist_ok=True)
            model.save_pretrained(save_path + '-clean')
            tokenizer.save_pretrained(save_path + '-clean')


def clean_model_train(args, model, tokenizer, dataloader_train, dataloader_dev, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model=True, save_path=None, save_metric='loss', eval_metric='acc',
                      freeze=False, clean=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    model = model.to(device)
    warm_epoch = 0

    for epoch in range(warm_epoch + epochs):

        train_loss, train_acc= train_embedding_l2(args, model, tokenizer, dataloader_train,
                                                             batch_size, optimizer, criterion, device, freeze)

        valid_loss, valid_acc = evaluate(model, tokenizer, dataloader_dev,
                                             batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
