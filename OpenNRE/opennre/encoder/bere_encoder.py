import math, logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from nltk import pos_tag
from . import bere_utils as utils
from ..tokenization import WordTokenizer


class BEREEncoder(nn.Module):

    def __init__(self,
                 token2id,
                 tag2id,
                 max_length=128,
                 hidden_size=230,
                 word_size=200,
                 tag_size=50,
                 blank_padding=True,
                 word2vec=None,
                 bidirectional=True,
                 dropout=0,
                 activation_function=F.relu,
                 mask_entity=False):
        """
        Args:
            token2id: dictionary of token->idx mapping
            tag2id: dictionary of tag->idx mapping
            max_length: max length of sentence
            hidden_size: hidden size
            word_size: size of word embedding
            tag_size: size of tag embedding
            blank_padding: padding for RNN
            word2vec: pretrained word2vec numpy
            bidirectional: if it is a bidirectional RNN
            activation_function: the activation function of RNN, tanh/relu
        """
        # Hyperparameters
        super(BEREEncoder, self).__init__()

        self.token2id = token2id
        self.tag2id = tag2id
        self.max_length = max_length
        self.num_token = len(token2id)
        self.num_tag = len(tag2id)
        self.num_position = max_length * 2
        self.bidirectional = bidirectional
        self.mask_entity = mask_entity

        if word2vec is None:
            self.word_size = word_size
        else:
            self.word_size = word2vec.shape[-1]
        self.tag_size = tag_size

        self.hidden_size = hidden_size
        self.input_size = word_size + tag_size
        self.blank_padding = blank_padding

        # add <head> and <tail> tokens
        if not '<head>' in self.token2id:
            self.token2id['<head>'] = len(self.token2id)
            self.num_token += 1
        if not '<head>' in self.tag2id:
            self.tag2id['<head>'] = len(self.tag2id)
            self.num_tag += 1
        if not '<tail>' in self.token2id:
            self.token2id['<tail>'] = len(self.token2id)
            self.num_token += 1
        if not '<tail>' in self.tag2id:
            self.tag2id['<tail>'] = len(self.tag2id)
            self.num_tag += 1

        # add [UNK] and [PAD] tokens
        if not '[UNK]' in self.token2id:
            self.token2id['[UNK]'] = len(self.token2id)
            self.num_token += 1
        if not '[UNK]' in self.tag2id:
            self.tag2id['[UNK]'] = len(self.tag2id)
            self.num_tag += 1
        if not '[PAD]' in self.token2id:
            self.token2id['[PAD]'] = len(self.token2id)
            self.num_token += 1
        if not '[PAD]' in self.tag2id:
            self.tag2id['[PAD]'] = len(self.tag2id)
            self.num_tag += 1

        # Word embedding
        self.word_embedding = nn.Embedding(self.num_token, self.word_size)
        if word2vec is not None:
            logging.info("Initializing word embedding with word2vec.")
            word2vec = torch.from_numpy(word2vec)
            if self.num_token == len(word2vec) + 4:  # 4 == <head>, <tail>, [UNK], [PAD]
                head = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                tail = torch.randn(1, self.word_size) / math.sqrt(self.word_size)

                unk = torch.randn(1, self.word_size) / math.sqrt(self.word_size)
                pad = torch.zeros(1, self.word_size)

                self.word_embedding.weight.data.copy_(torch.cat([word2vec, head, tail, unk, pad], 0))
            else:
                self.word_embedding.weight.data.copy_(word2vec)

        # Tag embedding
        self.tag_embedding = nn.Embedding(self.num_tag, self.tag_size)

        self.tokenizer = WordTokenizer(vocab=self.token2id, unk_token="[UNK]", tag_vocab=self.tag2id)

        self.drop = nn.Dropout(dropout)
        self.act = activation_function

        self.att = MultiAttn(self.input_size)
        self.leaf_rnn = LeafRNN(self.input_size, self.hidden_size, bidirectional=bidirectional)
        if bidirectional:
            self.hidden_size = 2 * self.hidden_size
        self.gumbel_rnn = GumbelTreeGRU(self.hidden_size)
        self.hidden_size = 3 * self.hidden_size
        # self.fc = nn.Linear(self.hidden_size, self.hidden_size // 10)
        # self.hidden_size = self.hidden_size // 10

    def forward(self, token, tag):
        """
        Args:
            token: (B, L), index of tokens
            tag: (B, L), index of tags
        Return:
            (B, H), representations for sentences
        """
        # Check size of tensors
        if len(token.size()) != 2 or token.size() != tag.size():
            raise Exception("Size of token and tag should be (B, L)")

        head_mask = torch.eq(token, self.token2id['<head>']).unsqueeze(-1).float()
        tail_mask = torch.eq(token, self.token2id['<tail>']).unsqueeze(-1).float()

        # Get non padding mask and sentence lengths (B,)
        non_pad_mask, length = utils.non_padding_mask(token, self.token2id['[PAD]'])
        # Get attention mask
        att_mask = utils.padding_mask(token, self.token2id['[PAD]'])

        word_emb = self.drop(self.word_embedding(token)) # (B, L, WORD_EMBED)
        tag_emb = self.drop(self.tag_embedding(tag))  # (B, L, TAG_EMBED)

        x = torch.cat([word_emb, tag_emb], dim=-1)
        x, word_att = self.att(x, att_mask, non_pad_mask)
        x = self.leaf_rnn(x, non_pad_mask, length)
        tree_feat, tree_order = self.gumbel_rnn(x, length)
        head_feat = (x * head_mask).sum(1)  # (B,D)
        tail_feat = (x * tail_mask).sum(1)  # (B,D)
        out = torch.cat([tree_feat, head_feat, tail_feat], -1)  # (B, 3D)
        # out = self.act(self.fc(feat))
        # out = self.drop(out)
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
            # ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
            # ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])

            # if self.mask_entity:
                # ent_0 = ['[UNK]']
                # ent_1 = ['[UNK]']

            if rev:
                ent_0 = ['<tail>']  # + ent_0 + ['</tail>']
                ent_1 = ['<head>']  # + ent_1 + ['</head>']
            else:
                ent_0 = ['<head>']  # + ent_0 + ['</head>']
                ent_1 = ['<tail>']  # + ent_1 + ['</tail>']

            tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
        else:
            tokens = sentence

        # Token -> tag
        tags = [tag[1] for tag in pos_tag(tokens)]

        # Token, Tag -> index
        if self.blank_padding:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'], self.token2id['[UNK]'])
            indexed_tags = self.tokenizer.convert_tags_to_ids(tags, self.max_length, self.tag2id['[PAD]'], self.tag2id['[UNK]'])
        else:
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id = self.token2id['[UNK]'])
            indexed_tags = self.tokenizer.convert_tags_to_ids(tags, unk_id= self.tag2id['[UNK]'])

        if self.blank_padding:
            indexed_tokens = indexed_tokens[:self.max_length]
            indexed_tags = indexed_tags[:self.max_length]

        indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0) # (1, L)
        indexed_tags = torch.tensor(indexed_tags).long().unsqueeze(0) # (1, L)

        return indexed_tokens, indexed_tags


class MultiAttn(nn.Module):
    def __init__(self, in_dim, head_num=10):
        super(MultiAttn, self).__init__()

        self.head_dim = in_dim // head_num
        self.head_num = head_num

        # scaled dot product attention
        self.scale = self.head_dim ** -0.5

        self.w_qs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
        self.w_ks = nn.Linear(in_dim, head_num * self.head_dim, bias=True)
        self.w_vs = nn.Linear(in_dim, head_num * self.head_dim, bias=True)

        self.w_os = nn.Linear(head_num * self.head_dim, in_dim, bias=True)

        self.gamma = nn.Parameter(torch.FloatTensor([0]))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, attn_mask, non_pad_mask):
        B, L, H = x.size()
        head_num = self.head_num
        head_dim = self.head_dim

        q = self.w_qs(x).view(B * head_num, L, head_dim)
        k = self.w_ks(x).view(B * head_num, L, head_dim)
        v = self.w_vs(x).view(B * head_num, L, head_dim)

        attn_mask = attn_mask.repeat(head_num, 1, 1)

        attn = torch.bmm(q, k.transpose(1, 2))  # B*head_num, L, L
        attn = self.scale * attn
        attn = attn.masked_fill_(attn_mask, -np.inf)
        attn = self.softmax(attn)

        out = torch.bmm(attn, v)  # B*head_num, L, head_dim

        out = out.view(B, L, head_dim * head_num)

        out = self.w_os(out)

        out = non_pad_mask * out

        out = self.gamma * out + x

        return out, attn


class LeafRNN(nn.Module):
    def __init__(self, in_dim, hid_dim, bidirectional=True):
        super(LeafRNN, self).__init__()
        self.bidirectional = bidirectional

        self.leaf_rnn = nn.GRU(in_dim, hid_dim, batch_first=True)

        if self.bidirectional:
            self.leaf_rnn_bw = nn.GRU(in_dim, hid_dim, batch_first=True)

    def forward(self, x, non_pad_mask, length=None):
        out, _ = self.leaf_rnn(x)
        out = non_pad_mask * out

        if self.bidirectional:
            in_bw = utils.reverse_padded_sequence(x, length, batch_first=True)
            out_bw, _ = self.leaf_rnn_bw(in_bw)
            out_bw = non_pad_mask * out_bw
            out_bw = utils.reverse_padded_sequence(out_bw, length, batch_first=True)
            out = torch.cat([out, out_bw], -1)

        return out


class BinaryTreeGRULayer(nn.Module):
    def __init__(self, hidden_dim):
        super(BinaryTreeGRULayer, self).__init__()

        self.fc1 = nn.Linear(in_features=2 * hidden_dim, out_features=3 * hidden_dim)
        self.fc2 = nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim)

    def forward(self, hl, hr):
        """
        Args:
            hl: (batch_size, max_length, hidden_dim).
            hr: (batch_size, max_length, hidden_dim).
        Returns:
            h: (batch_size, max_length, hidden_dim).
        """

        hlr_cat1 = torch.cat([hl, hr], dim=-1)
        treegru_vector = self.fc1(hlr_cat1)
        i, f, r = treegru_vector.chunk(chunks=3, dim=-1)

        hlr_cat2 = torch.cat([hl * r.sigmoid(), hr * r.sigmoid()], dim=-1)

        h_hat = self.fc2(hlr_cat2)

        h = (hl + hr) * f.sigmoid() + h_hat.tanh() * i.sigmoid()

        return h


class GumbelTreeGRU(nn.Module):
    def __init__(self, hidden_dim):
        super(GumbelTreeGRU, self).__init__()
        self.hidden_dim = hidden_dim

        self.gumbel_temperature = nn.Parameter(torch.FloatTensor([1]))

        self.treegru_layer = BinaryTreeGRULayer(hidden_dim)

        self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
        init.normal_(self.comp_query.data, mean=0, std=0.01)

        self.query_layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 10, bias=True), nn.Tanh(),
                                         nn.Linear(hidden_dim // 10, 1, bias=True))

    @staticmethod
    def update_state(old_h, new_h, done_mask):
        done_mask = done_mask.float().unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        return h

    def select_composition(self, old_h, new_h, mask):
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]

        comp_weights = self.query_layer(new_h).squeeze(2)


        if self.training:
            select_mask = utils.st_gumbel_softmax(
                logits=comp_weights, temperature=self.gumbel_temperature,
                mask=mask)
        else:
            select_mask = utils.greedy_select(logits=comp_weights, mask=mask).float()

        select_mask_cumsum = select_mask.cumsum(1)
        left_mask = 1 - select_mask_cumsum
        right_mask = select_mask_cumsum - select_mask

        new_h = (select_mask.unsqueeze(2) * new_h
                 + left_mask.unsqueeze(2) * old_h_left
                 + right_mask.unsqueeze(2) * old_h_right)

        return new_h, select_mask

    def forward(self, input, length):
        max_depth = input.size(1)
        length_mask = utils.sequence_mask(length=length, max_length=max_depth)
        select_masks = []

        h = input

        for i in range(max_depth - 1):
            hl = h[:, :-1, :]
            hr = h[:, 1:, :]
            new_h = self.treegru_layer(hl, hr)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, select_mask = self.select_composition(
                    old_h=h, new_h=new_h,
                    mask=length_mask[:, i + 1:])

                select_masks.append(select_mask)

            done_mask = length_mask[:, i + 1]

            h = self.update_state(old_h=h, new_h=new_h,
                                  done_mask=done_mask)

        out = h.squeeze(1)

        return out, select_masks
