import argparse
import os
import random
import math

import numpy as np
import torch
import pandas as pd
import pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset
from pytorch_transformers import *
from torch.autograd import Variable
from torch.utils.data import Dataset
from transformers.modeling_bert import BertEmbeddings, BertPooler, BertLayer

from tqdm import tqdm

class BertModel4Mix(BertPreTrainedModel):
    def __init__(self, config):
        super(BertModel4Mix, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder4Mix(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(
            old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids,  input_ids2=None, l=None, mix_layer=1000, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):

        if attention_mask is None:
            if input_ids2 is not None:
                attention_mask2 = torch.ones_like(input_ids2)
            attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            if input_ids2 is not None:
                token_type_ids2 = torch.zeros_like(input_ids2)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if input_ids2 is not None:

            extended_attention_mask2 = attention_mask2.unsqueeze(
                1).unsqueeze(2)

            extended_attention_mask2 = extended_attention_mask2.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask2 = (
                1.0 - extended_attention_mask2) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(
                    0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(
                    self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                # We can specify head_mask for each layer
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
            # switch to fload if need + fp16 compatibility
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if input_ids2 is not None:
            embedding_output2 = self.embeddings(
                input_ids2, position_ids=position_ids, token_type_ids=token_type_ids2)

        if input_ids2 is not None:
            encoder_outputs = self.encoder(embedding_output, embedding_output2, l, mix_layer,
                                           extended_attention_mask, extended_attention_mask2, head_mask=head_mask)
        else:
            encoder_outputs = self.encoder(
                embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask)

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # add hidden_states and attentions if they are here
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return outputs


class BertEncoder4Mix(nn.Module):
    def __init__(self, config):
        super(BertEncoder4Mix, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, hidden_states2=None, l=None, mix_layer=1000, attention_mask=None, attention_mask2=None, head_mask=None):
        all_hidden_states = ()
        all_attentions = ()

        # Perform mix at till the mix_layer
        if mix_layer == -1:
            if hidden_states2 is not None:
                hidden_states = l * hidden_states + (1-l)*hidden_states2

        for i, layer_module in enumerate(self.layer):
            if i <= mix_layer:

                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

                if hidden_states2 is not None:
                    layer_outputs2 = layer_module(
                        hidden_states2, attention_mask2, head_mask[i])
                    hidden_states2 = layer_outputs2[0]

            if i == mix_layer:
                if hidden_states2 is not None:
                    hidden_states = l * hidden_states + (1-l)*hidden_states2

            if i > mix_layer:
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i])
                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        # last-layer hidden state, (all hidden states), (all attentions)
        return outputs


class MixText(nn.Module):
    def __init__(self, num_labels=2, mix_option=False):
        super(MixText, self).__init__()

        if mix_option:
            self.bert = BertModel4Mix.from_pretrained('bert-base-uncased')
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.linear = nn.Sequential(nn.Linear(768, 128),
                                    nn.Tanh(),
                                    nn.Linear(128, num_labels))

    def forward(self, x, x2=None, l=None, mix_layer=1000):

        if x2 is not None:
            all_hidden, pooler = self.bert(x, x2, l, mix_layer)

            pooled_output = torch.mean(all_hidden, 1)

        else:
            all_hidden, pooler = self.bert(x)

            pooled_output = torch.mean(all_hidden, 1)

        predict = self.linear(pooled_output)

        return predict

class Translator:
    """Backtranslation. Here to save time, we pre-processing and save all the translated data into pickle files.
    """

    def __init__(self, path, transform_type='BackTranslation'):
        # Pre-processed German data
        with open(path + 'de_1.pkl', 'rb') as f:
            self.de = pickle.load(f)
        # Pre-processed Russian data
        with open(path + 'ru_1.pkl', 'rb') as f:
            self.ru = pickle.load(f)

    def __call__(self, ori, idx):
        if (idx in self.de):
            out1 = self.de[idx]
            out2 = self.ru[idx]
            return out1, out2, ori
        return ori, ori, ori

def get_data(data_path, n_labeled_per_class, unlabeled_per_class=5000, max_seq_len=256):
    
    model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model)
    train_df = pd.read_csv(data_path+'train.csv', header=None)
    test_df = pd.read_csv(data_path+'test.csv', header=None)
    # Here we only use the bodies and removed titles to do the classifications
    train_labels = np.array([train_df[0][i] - 1 for i in range(200000)])  # [v-1 for v in train_df[0]]
    train_text = np.array([train_df[2][i] for i in range(200000)])  # ([v for v in train_df[2]])

    test_labels = np.array([u-1 for u in test_df[0]])
    test_text = np.array([v for v in test_df[2]])

    n_labels = max(test_labels) + 1

    np.random.seed(0)
    labels = np.array(train_labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(n_labels):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_pool = idxs[5000:-5000]
        train_labeled_idxs.extend(train_pool[:n_labeled_per_class])
        train_unlabeled_idxs.extend(
            idxs[:5000])
        val_idxs.extend(idxs[-5000:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    train_labeled_dataset = loader_labeled(
        train_text[train_labeled_idxs], train_labels[train_labeled_idxs], tokenizer, max_seq_len)
    train_unlabeled_dataset = loader_unlabeled(
        train_text[train_unlabeled_idxs], train_unlabeled_idxs, tokenizer, max_seq_len, Translator(data_path))
    val_dataset = loader_labeled(
        train_text[val_idxs], train_labels[val_idxs], tokenizer, max_seq_len)
    test_dataset = loader_labeled(
        test_text, test_labels, tokenizer, max_seq_len)

    print("#Labeled: {}, Unlabeled {}, Val {}, Test {}".format(len(
        train_labeled_idxs), len(train_unlabeled_idxs), len(val_idxs), len(test_labels)))

    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels

class loader_labeled(Dataset):
    def __init__(self, dataset_text, dataset_label, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.labels = dataset_label
        self.max_seq_len = max_seq_len
        self.trans_dist = {}
        self.aug = False

    def __len__(self):
        return len(self.labels)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)

        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding

        return encode_result, length

    def __getitem__(self, idx):
        if self.aug:
            text = self.text[idx]
            text_aug = self.augment(text)
            text_result, text_length = self.get_tokenized(text)
            text_result2, text_length2 = self.get_tokenized(text_aug)
            return ((torch.tensor(text_result), torch.tensor(text_result2)), (self.labels[idx], self.labels[idx]), (text_length, text_length2))
        else:
            text = self.text[idx]
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) > self.max_seq_len:
                tokens = tokens[:self.max_seq_len]
            length = len(tokens)
            encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
            padding = [0] * (self.max_seq_len - len(encode_result))
            encode_result += padding
            return (torch.tensor(encode_result), self.labels[idx], length)


class loader_unlabeled(Dataset):
    # Data loader for unlabeled data
    def __init__(self, dataset_text, unlabeled_idxs, tokenizer, max_seq_len, aug=None):
        self.tokenizer = tokenizer
        self.text = dataset_text
        self.ids = unlabeled_idxs
        self.aug = aug
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.text)

    def get_tokenized(self, text):
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        length = len(tokens)
        encode_result = self.tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.max_seq_len - len(encode_result))
        encode_result += padding
        return encode_result, length

    def __getitem__(self, idx):
        if self.aug is not None:
            u, v, ori = self.aug(self.text[idx], self.ids[idx])
            encode_result_u, length_u = self.get_tokenized(u)
            encode_result_v, length_v = self.get_tokenized(v)
            encode_result_ori, length_ori = self.get_tokenized(ori)
            return ((torch.tensor(encode_result_u), torch.tensor(encode_result_v), torch.tensor(encode_result_ori)), (length_u, length_v, length_ori),idx)
        else:
            text = self.text[idx]
            encode_result, length = self.get_tokenized(text)
            return (torch.tensor(encode_result), length)


parser = argparse.ArgumentParser(description='PyTorch MixText')

parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--batch-size-u', default=24, type=int, metavar='N',
                    help='train batchsize')

parser.add_argument('--lrmain', '--learning-rate-bert', default=0.00001, type=float,
                    metavar='LR', help='initial learning rate for bert')
parser.add_argument('--lrlast', '--learning-rate-model', default=0.001, type=float,
                    metavar='LR', help='initial learning rate for models')

parser.add_argument('--n-labeled', type=int, default=20,
                    help='number of labeled data')
parser.add_argument('--un-labeled', default=5000, type=int,
                    help='number of unlabeled data')
parser.add_argument('--val-iteration', type=int, default=200,
                    help='number of labeled data')

parser.add_argument('--data-path', type=str, default='yahoo_answers_csv/',
                    help='path to data folders')

parser.add_argument('--alpha', default=0.75, type=float,
                    help='alpha for beta distribution')



args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print("GPU num: ", n_gpu)

best_acc = 0
total_steps = 0
flag = 0

de_flowgmm_lbls ={}
ru_flowgmm_lbls ={}
ori_flowgmm_lbls ={}

def main():
    global best_acc
    global de_flowgmm_lbls
    global ru_flowgmm_lbls
    global ori_flowgmm_lbls
    dat_p ="../../input/flowgmm-labels-for-yahoo/"
    with open(dat_p + 'de_flowgmm_labels.pkl', 'rb') as f:
        de_flowgmm_lbls = pickle.load(f)
    with open(dat_p  + 'ru_flowgmm_labels.pkl', 'rb') as f:
        ru_flowgmm_lbls = pickle.load(f)
    with open(dat_p  + 'ori_flowgmm_labels.pkl', 'rb') as f:
        ori_flowgmm_lbls = pickle.load(f)

    # Read dataset and build dataloaders
    train_labeled_set, train_unlabeled_set, val_set, test_set, n_labels = get_data(
        args.data_path, args.n_labeled, args.un_labeled)
    labeled_trainloader = Data.DataLoader(
        dataset=train_labeled_set, batch_size=args.batch_size, shuffle=True)
    unlabeled_trainloader = Data.DataLoader(
        dataset=train_unlabeled_set, batch_size=args.batch_size_u, shuffle=True)
    val_loader = Data.DataLoader(
        dataset=val_set, batch_size=512, shuffle=False)
    test_loader = Data.DataLoader(
        dataset=test_set, batch_size=512, shuffle=False)

    # Define the model, set the optimizer
    model = MixText(n_labels, True).cuda()
    model = nn.DataParallel(model)


    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()

    test_accs = []

    optimizer = AdamW(
        [
            {"params": model.module.bert.parameters(), "lr": args.lrmain},
            {"params": model.module.linear.parameters(), "lr": args.lrlast},
        ])
    # Start training
    for epoch in range(args.epochs):

        train(labeled_trainloader, unlabeled_trainloader, model, optimizer, train_criterion, epoch, n_labels)

        val_loss, val_acc = validate(
            val_loader, model, criterion, epoch, mode='Valid Stats')

        print("epoch {}, val acc {}, val_loss {}".format(
            epoch, val_acc, val_loss))

        if val_acc >= best_acc:
            best_acc = val_acc
            test_loss, test_acc = validate(
                test_loader, model, criterion, epoch, mode='Test Stats ')
            test_accs.append(test_acc)
            print("epoch {}, test acc {},test loss {}".format(
                epoch, test_acc, test_loss))

        print('Epoch: ', epoch)

        print('Best acc:')
        print(best_acc)

        print('Test acc:')
        print(test_accs)

    print("Finished training!")
    print('Best acc:')
    print(best_acc)

    print('Test acc:')
    print(test_accs)


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, criterion, epoch, n_labels):
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    model.train()

    global de_flowgmm_lbls
    global ru_flowgmm_lbls
    global ori_flowgmm_lbls

    global total_steps
    global flag
    if flag == 0 and total_steps > 1000000:
        T = 0.9
        flag = 1
    else:
        T =0.5
    for batch_idx in tqdm(range(args.val_iteration)):

        total_steps += 1

        inputs_x, targets_x, inputs_x_length = labeled_train_iter.next()

        (inputs_u, inputs_u2, inputs_ori), (length_u,
                                            length_u2, length_ori), u_idxs = unlabeled_train_iter.next()

        batch_size = inputs_x.size(0)
        batch_size_2 = inputs_ori.size(0)
        targets_x = torch.zeros(batch_size, n_labels).scatter_(
            1, targets_x.view(-1, 1), 1)

        print("sizes",inputs_x.shape, inputs_u.shape,inputs_u2.shape,inputs_ori.shape, u_idxs)

        inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
        inputs_u = inputs_u.cuda()
        inputs_u2 = inputs_u2.cuda()
        inputs_ori = inputs_ori.cuda()

        out_u = [(de_flowgmm_lbls[idx.item()] if idx.item() in de_flowgmm_lbls else random.randint(0,9))  for idx in
                 u_idxs]
        out_u2 = [(ru_flowgmm_lbls[idx.item()] if idx.item() in ru_flowgmm_lbls else random.randint(0, 9)) for idx in
                 u_idxs]
        out_ori = [(ori_flowgmm_lbls[idx.item()] if idx.item() in ori_flowgmm_lbls else random.randint(0, 9)) for idx in
                 u_idxs]

        for idx in u_idxs:
            print(idx.item())
        print("Labels", len(de_flowgmm_lbls),de_flowgmm_lbls[420562], out_u)
        mask = []

        with torch.no_grad():
            # Predict labels for unlabeled data.
            #outputs_u = model(inputs_u)
            #outputs_u2 = model(inputs_u2)
            #outputs_ori = model(inputs_ori)
            #print("output:",type(outputs_u), outputs_u.shape, outputs_u)
            outputs = [[(1 if j ==i else 0) for j in range(10)] for i in out_u]
            outputs = torch.FloatTensor(outputs)
            print("output u:",outputs)
            outputs_u = outputs.cuda()

            outputs = [[(1 if j == i else 0) for j in range(10)] for i in out_u2]
            outputs = torch.FloatTensor(outputs)
            print("output u2:",outputs)
            outputs_u2 = outputs.cuda()

            outputs = [[(1 if j == i else 0) for j in range(10)] for i in out_ori]
            outputs = torch.FloatTensor(outputs)
            print("output ori:",outputs)
            outputs_ori = outputs.cuda()


            p = (1 * torch.softmax(outputs_u, dim=1)
                 + 0 * torch.softmax(outputs_u2,dim=1) + 1 * torch.softmax(outputs_ori, dim=1)) / 2
            pt = p**(1/T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        mixed = 1

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        mixl = [7,9,12]
        mix_layer = np.random.choice(mixl, 1)[0]
        mix_layer = mix_layer - 1

        all_inputs = torch.cat(
                [inputs_x, inputs_u, inputs_u2, inputs_ori, inputs_ori], dim=0)

        all_lengths = torch.cat(
                [inputs_x_length, length_u, length_u2, length_ori, length_ori], dim=0)

        all_targets = torch.cat(
                [targets_x, targets_u, targets_u, targets_u, targets_u], dim=0)

        print("all inputs size:",all_inputs.shape,all_lengths.shape,all_targets.shape)

        idx1 = torch.randperm(all_inputs.size(0) - batch_size_2)
        idx2 = torch.arange(batch_size_2) + all_inputs.size(0) - batch_size_2
        idx = torch.cat([idx1, idx2], dim=0)

        print("Indexes  are",idx1,idx2,idx)
        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        length_a, length_b = all_lengths, all_lengths[idx]
        print("input a,b",input_a.shape,input_b.shape)


        logits = model(input_a, input_b, l, mix_layer)
        mixed_target = l * target_a + (1 - l) * target_b

        Lx, Lu, w, Lu2, w2 = criterion(logits[:batch_size], mixed_target[:batch_size], logits[batch_size:-batch_size_2],
                                       mixed_target[batch_size:-batch_size_2], logits[-batch_size_2:], epoch+batch_idx/args.val_iteration, mixed)

        loss = Lx + w * Lu

        #max_grad_norm = 1.0
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if batch_idx % 1000 == 0:
            print("epoch {}, step {}, loss {}, Lx {}, Lu {}, Lu2 {}".format(
                epoch, batch_idx, loss.item(), Lx.item(), Lu.item(), Lu2.item()))


def validate(valloader, model, criterion, epoch, mode):
    model.eval()
    with torch.no_grad():
        loss_total = 0
        total_sample = 0
        acc_total = 0
        correct = 0

        for batch_idx, (inputs, targets, length) in enumerate(valloader):
            inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            _, predicted = torch.max(outputs.data, 1)

            if batch_idx == 0:
                print("Sample some true labeles and predicted labels")
                print(predicted[:20])
                print(targets[:20])

            correct += (np.array(predicted.cpu()) ==
                        np.array(targets.cpu())).sum()
            loss_total += loss.item() * inputs.shape[0]
            total_sample += inputs.shape[0]

        acc_total = correct/total_sample
        loss_total = loss_total/total_sample

    return loss_total, acc_total


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, outputs_u_2, epoch, mixed=1):

            Lx = - \
                torch.mean(torch.sum(F.log_softmax(
                    outputs_x, dim=1) * targets_x, dim=1))

            probs_u = torch.softmax(outputs_u, dim=1)

            Lu = F.kl_div(probs_u.log(), targets_u, None, None, 'batchmean')

            Lu2 = torch.mean(torch.clamp(torch.sum(-F.softmax(outputs_u, dim=1)
                                                   * F.log_softmax(outputs_u, dim=1), dim=1) - 0.7, min=0))

            return Lx, Lu, linear_rampup(epoch), Lu2, linear_rampup(epoch)


if __name__ == '__main__':
    main()
