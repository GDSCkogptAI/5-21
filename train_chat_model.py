import argparse
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader, Dataset
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
from dataloader import fundchatdataset

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = "</s>"
EOS = "</s>"
PAD = "<pad>"
MASK = "<unused0>"
SENT = "<unused1>"

data_path = 'C:/kogpt2/ChatBotData.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    data = [item[0] for item in batch]
    mask = [item[1] for item in batch]
    label = [item[2] for item in batch]
    return torch.LongTensor(data), torch.LongTensor(mask), torch.LongTensor(label)


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", default = 5, type = int)
#parser.add_argument("--lr", default = 3e-5, type = float)
parser.add_argument("--batch_size", default = 32, type = int)
parser.add_argument("--warmup_steps", default = 100, type = int)
args = parser.parse_args('')

tokenizer = PreTrainedTokenizerFast.from_pretrained('skt/kogpt2-base-v2', bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

train_data = pd.read_csv(data_path)
#train_data = train_data.head(1000)

train_set = fundchatdataset(train_data, max_len = 20)
train_dataloader = DataLoader(train_set, batch_size = args.batch_size, num_workers = 0, shuffle = True, collate_fn = collate_batch,)

model.to(device)
model.train()

criterion = torch.nn.CrossEntropyLoss(reduction = "none").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 5e-5)
Sneg = -1e18

print(f"epoch = {args.epochs}")
print("start")
for epoch in range(args.epochs):
    print(f"Training epoch {epoch}")
    for batch_idx, samples in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids, mask, label = samples
        token_ids = token_ids.to(device)
        mask = mask.to(device)
        label = label.to(device)
        out = model(token_ids)
        out = out.logits
        mask_3d = mask.unsqueeze(dim = 2).repeat_interleave(repeats = out.shape[2], dim = 2).to(device)
        mask_out = torch.where(mask_3d == 1, out, Sneg * torch.ones_like(out)).to(device)
        loss = criterion(mask_out.transpose(1, 2), label).to(device)
        avg_loss = loss.sum() / mask.sum()
        avg_loss.backward()
        optimizer.step()
        #scheduler.step()
        #if batch_idx % 100 == 0:
            #print("Batch: {}, Loss: {}, LR: {}".format(batch_idx, avg_loss, scheduler.get_last_lr()[0]))
    print("Epoch: {}, Loss: {}".format(epoch, avg_loss))
    model.save_pretrained("./chat_model")
    tokenizer.save_pretrained("./chat_model")




