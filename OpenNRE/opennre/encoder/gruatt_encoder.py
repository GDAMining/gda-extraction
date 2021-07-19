import math, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..tokenization import WordTokenizer

class GRUAttEncoder(nn.Module):

    def __init__(self,
                 token2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=50,
                 blank_padding=True,
                 word2vec=None,
                 bidirectional=True,
                 dropout=0,
                 activation_function=F.tanh,
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            max_length: max length of sentence
            hidden_size: hidden size
            word_size: size of word embedding
            blank_padding: padding for RNN
            word2vec: pretrained word2vec numpy
            bidirectional: if it is a bidirectional RNN
            activation_function: the activation function of RNN, tanh/relu
        """
        # Hyperparameters
        super(GRUAttEncoder, self).__init__()

        self.token2id = token2id
        self.max_length = max_length + 4  # 4 == take into account PIs
        self.num_token = len(token2id)
        self.num_position = max_length * 2
        self.bidirectional = bidirectional
        self.mask_entity = mask_entity

        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]

        self.hidden_size = hidden_size
        self.input_size = word_size
        self.blank_padding = blank_padding

        # Position Indicators (PI)
        if not '<head>' in self.token2id:
            self.token2id['<head>'] = len(self.token2id)
            self.num_token += 1
        if not '</head>' in self.token2id:
            self.token2id['</head>'] = len(self.token2id)
            self.num_token += 1
        if not '<tail>' in self.token2id:
            self.token2id['<tail>'] = len(self.token2id)
            self.num_token += 1
        if not '</tail>' in self.token2id:
            self.token2id['</tail>'] = len(self.token2id)
            self.num_token += 1

        # add [UNK] and [PAD] tokens
        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1

        # Word embedding
        self.word_embedding = nn.Embedding(self.num_token, self.word_size)
        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 6:  # 6 == <head>, </head>, <tail>, </tail>, [UNK], [PAD]
                hsp = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                hep = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                tsp = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                tep = torch.randn(1, self.word_size) / math.sqrt(self.word_size)

                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                pad = torch.zeros(1, self.word_size)

                self.word_embedding.weight.data.copy_(torch.cat([word2vec, hsp, hep, tsp, tep, unk, pad], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)

        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]")

        self.drop = nn.Dropout(dropout)
        self.act = activation_function
        self.att = Attention(self.hidden_size)

        self.gru_fw = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        if self.bidirectional:
            self.gru_bw = nn.GRU(self.input_size, self.hidden_size, batch_first=True)

    def forward(self, token):
        """
        Args:
            token: (B, L), index of tokens
        Return:
            (B, H), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2:
            raise Exception("Size of token should be (B, L)")

        # Get non padding mask and sentence lengths (B,)
        non_pad_mask, length = self.non_padding_mask(token)
        # Get attention mask
        att_mask = self.padding_mask(token)

        x = self.word_embedding(token) # (B, L, EMBED)

        out_fw, _ = self.gru_fw(x)
        out = non_pad_mask * out_fw
        if self.bidirectional:
            x_bw = self.reverse_padded_sequence(x, length, batch_first=True)
            out_bw, _ = self.gru_bw(x_bw)
            out_bw = non_pad_mask * out_bw
            out_bw = self.reverse_padded_sequence(out_bw, length, batch_first=True)
            out = torch.add(out, out_bw) # (B, L, H)

        out, _ = self.att(out, att_mask)
        out = self.drop(out)
        return out

    def tokenize(self, item):
        """
        Args:
            item: input instance, including sentence, entity positions, etc.
        Return:
            index number of tokens and positions
        """
        if 'text' in item:
            sentence = item['text']
            is_token = False
        else:
            sentence = item['token']
            is_token = True
        pos_head = item['h']['pos']
        pos_tail = item['t']['pos']

        # Sentence -> token
        if not is_token:
            if pos_head[0] > pos_tail[0]:
                pos_min, pos_max = [pos_tail, pos_head]
                rev = True
            else:
                pos_min, pos_max = [pos_head, pos_tail]
                rev = False
            sent_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
            sent_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
            sent_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
            ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])

            if self.mask_entity:
                ent_0 = ['[UNK]']
                ent_1 = ['[UNK]']

            if rev:
                ent_0 = ['<tail>'] + ent_0 + ['</tail>']
                ent_1 = ['<head>'] + ent_1 + ['</head>']
            else:
                ent_0 = ['<head>'] + ent_0 + ['</head>']
                ent_1 = ['<tail>'] + ent_1 + ['</tail>']

            tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
        else:
            tokens = sentence

        # Token -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.token2id['[UNK]'])

        if self.blank_padding:
            indexed_tokens = indexed_tokens[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)

        return indexed_tokens

    def reverse_padded_sequence(self, x, lengths, batch_first=True):
        """Reverses sequences according to their lengths.
        Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
        ``B x T x *`` if True. T is the length of the longest sequence (or larger),
        B is the batch size, and * is any number of dimensions (including 0).
        Arguments:
            x (tensor): padded batch of variable length sequences.
            lengths (list[int]): list of sequence lengths
            batch_first (bool, optional): if True, inputs should be B x T x *.
        Returns:
            A tensor with the same size as inputs, but with each sequence
            reversed according to its length.
        """

        if not batch_first:
            x = x.transpose(0, 1)
        if x.size(0) != len(lengths):
            raise ValueError('inputs incompatible with lengths.')
        reversed_indices = [list(range(x.size(1))) for _ in range(x.size(0))]
        for i, length in enumerate(lengths):
            if length > 0:
                reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
        reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2).expand_as(x))
        reversed_indices = reversed_indices.to(x.device)
        reversed_x = torch.gather(x, 1, reversed_indices)
        if not batch_first:
            reversed_x = reversed_x.transpose(0, 1)
        return reversed_x

    def non_padding_mask(self, token):
        non_pad_mask = token.ne(self.token2id['[PAD]']).type(torch.float)
        length = torch.count_nonzero(non_pad_mask, dim=1)
        return non_pad_mask.unsqueeze(-1), length

    def padding_mask(self, token):
        pad_mask = token.eq(self.token2id['[PAD]'])
        return pad_mask


class Attention(nn.Module):
    """Applies dot-product attention"""

    def __init__(self, hidden_size):
        """
        Args:
            hidden_size: hidden size
        """
        # Hyperparameters
        super(Attention, self).__init__()

        self.u_omega = torch.nn.Parameter(torch.randn(1, hidden_size) / math.sqrt(hidden_size))
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = F.tanh

    def forward(self, x, att_mask):
        """
        Args:
            token: (B, L, H), index of tokens
        Return:
            (B, H), representations for sentences
        """
        v = self.tanh(x)
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = torch.matmul(v, self.u_omega.squeeze(0))  # (B,L) shape
        vu = vu.masked_fill_(att_mask, -np.inf)
        alpha = self.softmax(vu)  # (B,L) shape
        # x is reduced with attention vector; the result has (B,H) shape
        out = torch.sum(x * alpha.unsqueeze(-1), dim=1)
        # Apply tanh to output
        out = self.tanh(out)
        return out, alpha
