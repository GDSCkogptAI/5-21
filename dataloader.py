import math
import pandas as pd
import numpy as np
import random
import re
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerFast
import logging
import urllib.request

data_path = 'C:/kogpt2/ChatbotData.csv'  #chat_model

Chatbot_Data = pd.read_csv(data_path)

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"

tok_path = 'C:/kogpt2/tokenizer_with_custom_tokens'
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token = BOS, eos_token = EOS, unk_token = "<unk>", pad_token = PAD, mask_token = MASK)

class fundchatdataset(Dataset):
    def __init__(self, chats, max_len=20):
        self._data = chats
        self.max_len = max_len
        self.q_token = Q_TKN
        self.a_token = A_TKN
        self.sent_token = SENT
        self.bos = BOS
        self.eos = EOS
        self.mask = MASK
        self.tokenizer = tokenizer
        self.first = True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]
        q = turn["question"]
        q = re.sub(r"([?.!,])", r" ", q)

        a = turn["answer"]
        q_toked = self.tokenizer.tokenize(self.q_token + q + self.sent_token)
        q_len = len(q_toked)

        a_toked = self.tokenizer.tokenize(self.a_token + a + self.eos)
        a_len = len(a_toked)

        if q_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)) :]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
                assert a_len > 0
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)
            assert a_len == len(a_toked), f'{a_len} ==? {len(a_toked)}'

        if q_len + a_len > self.max_len:
            a_len = self.max_len - q_len
            if a_len <= 0:
                q_toked = q_toked[-(int(self.max_len / 2)) :]
                q_len = len(q_toked)
                a_len = self.max_len - q_len
            a_toked = a_toked[:a_len]
            a_len = len(a_toked)

        labels = [self.mask,] * q_len + a_toked[1:]
        if self.first:
            logging.info("contexts : {}".format(q))
            logging.info("toked ctx: {}".format(q_toked))
            logging.info("response : {}".format(a))
            logging.info("toked response : {}".format(a_toked))
            logging.info('labels {}'.format(labels))
            self.first = False

        mask = [0] * q_len + [1] * a_len + [0] * (self.max_len - q_len - a_len)
        # Attention mask 생성
        attention_mask = [1] * (q_len + a_len) + [0] * (self.max_len - q_len - a_len)

        self.max_len
        labels_ids = self.tokenizer.convert_tokens_to_ids(labels)
        while len(labels_ids) < self.max_len:
            labels_ids += [self.tokenizer.pad_token_id]

        token_ids = self.tokenizer.convert_tokens_to_ids(q_toked + a_toked)
        while len(token_ids) < self.max_len:
            token_ids += [self.tokenizer.pad_token_id]

        return (token_ids, np.array(attention_mask), labels_ids)
