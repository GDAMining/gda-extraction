import torch
from torch import nn, optim
from .base_model import BagRE


class BagCNNAverage(BagRE):
    """
    Average policy for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ..., 
                  'h': {'pos': [start, end], ...}, 
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        # pass

        self.eval()
        tokens = []
        pos1s = []
        pos2s = []
        masks = []
        for item in bag:
            token, pos1, pos2, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            pos1s.append(pos1)
            pos2s.append(pos2)
            masks.append(mask)
        tokens = torch.cat(tokens, 0) # (n, L)
        pos1s = torch.cat(pos1s, 0)
        pos2s = torch.cat(pos2s, 0)
        masks = torch.cat(masks, 0) 
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, pos1s, pos2s, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)
    
    def forward(self, label, scope, token, pos1, pos2, mask=None, train=True, bag_size=None):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            pos1: (nsum, L), relative position to head entity
            pos2: (nsum, L), relative position to tail entity
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """
        
        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            pos1 = pos1.view(-1, pos1.size(-1))
            pos2 = pos2.view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            pos1 = pos1[:, begin:end, :].view(-1, pos1.size(-1))
            pos2 = pos2[:, begin:end, :].view(-1, pos2.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        if mask is not None:
            rep = self.sentence_encoder(token, pos1, pos2, mask) # (nsum, H)
        else:
            rep = self.sentence_encoder(token, pos1, pos2) # (nsum, H)

        # Average
        if train:
            if bag_size == 0:
                bag_rep = []
                for i in range(len(scope)):
                    bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = label.size(0)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                bag_rep = rep.mean(1) # (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:
            if bag_size == 0:
                bag_rep = []
                for i in range(len(scope)):
                    bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
                bag_rep = torch.stack(bag_rep, 0) # (B, H) 
            else:
                batch_size = rep.size(0) // bag_size
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                bag_rep = rep.mean(1) # (B, H)
            bag_logits = self.softmax(self.fc(bag_rep)) # (B, N)
        return bag_logits


class BagGRUAverage(BagRE):
    """
    Average policy for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Linear(self.sentence_encoder.hidden_size, num_class)
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ...,
                  'h': {'pos': [start, end], ...},
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        # pass

        self.eval()
        tokens = []
        masks = []
        for item in bag:
            token, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            masks.append(mask)
        tokens = torch.cat(tokens, 0) # (n, L)
        masks = torch.cat(masks, 0)
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)

    def forward(self, label, scope, token, mask=None, train=True, bag_size=None):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """

        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        if mask is not None:
            rep = self.sentence_encoder(token, mask) # (nsum, H)
        else:
            rep = self.sentence_encoder(token) # (nsum, H)

        # Average
        if train:
            if bag_size == 0:
                bag_rep = []
                for i in range(len(scope)):
                    bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = label.size(0)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                bag_rep = rep.mean(1) # (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:
            if bag_size == 0:
                bag_rep = []
                for i in range(len(scope)):
                    bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = rep.size(0) // bag_size
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                bag_rep = rep.mean(1) # (B, H)
            bag_logits = self.softmax(self.fc(bag_rep)) # (B, N)
        return bag_logits


class BagBEREAverage(BagRE):
    """
    Average policy for bag-level relation extraction.
    """

    def __init__(self, sentence_encoder, num_class, rel2id):
        """
        Args:
            sentence_encoder: encoder for sentences
            num_class: number of classes
            id2rel: dictionary of id -> relation name mapping
        """
        super().__init__()
        self.sentence_encoder = sentence_encoder
        self.num_class = num_class
        self.fc = nn.Sequential(nn.Linear(self.sentence_encoder.hidden_size, self.sentence_encoder.hidden_size // 10),
                                nn.ReLU(),
                                nn.Linear(self.sentence_encoder.hidden_size // 10, num_class))
        self.softmax = nn.Softmax(-1)
        self.rel2id = rel2id
        self.id2rel = {}
        self.drop = nn.Dropout()
        for rel, id in rel2id.items():
            self.id2rel[id] = rel

    def infer(self, bag):
        """
        Args:
            bag: bag of sentences with the same entity pair
                [{
                  'text' or 'token': ...,
                  'h': {'pos': [start, end], ...},
                  't': {'pos': [start, end], ...}
                }]
        Return:
            (relation, score)
        """
        # pass

        self.eval()
        tokens = []
        tags = []
        masks = []
        for item in bag:
            token, tag, mask = self.sentence_encoder.tokenize(item)
            tokens.append(token)
            tags.append(tag)
            masks.append(mask)
        tokens = torch.cat(tokens, 0) # (n, L)
        tags = torch.cat(tags, 0)
        masks = torch.cat(masks, 0)
        scope = torch.tensor([[0, len(bag)]]).long() # (1, 2)
        bag_logits = self.forward(None, scope, tokens, tags, masks, train=False).squeeze(0) # (N) after softmax
        score, pred = bag_logits.max()
        score = score.item()
        pred = pred.item()
        rel = self.id2rel[pred]
        return (rel, score)

    def forward(self, label, scope, token, tag, mask=None, train=True, bag_size=None):
        """
        Args:
            label: (B), label of the bag
            scope: (B), scope for each bag
            token: (nsum, L), index of tokens
            tag: (nsum, L), index of tags
            mask: (nsum, L), used for piece-wise CNN
        Return:
            logits, (B, N)
        """

        if bag_size > 0:
            token = token.view(-1, token.size(-1))
            tag = tag.view(-1, tag.size(-1))
            if mask is not None:
                mask = mask.view(-1, mask.size(-1))
        else:
            begin, end = scope[0][0], scope[-1][1]
            token = token[:, begin:end, :].view(-1, token.size(-1))
            tag = tag[:, begin:end, :].view(-1, tag.size(-1))
            if mask is not None:
                mask = mask[:, begin:end, :].view(-1, mask.size(-1))
            scope = torch.sub(scope, torch.zeros_like(scope).fill_(begin))
        if mask is not None:
            rep = self.sentence_encoder(token, tag, mask) # (nsum, H)
        else:
            rep = self.sentence_encoder(token, tag) # (nsum, H)

        # Average
        if train:
            if bag_size == 0:
                bag_rep = []
                for i in range(len(scope)):
                    bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = label.size(0)
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                bag_rep = rep.mean(1) # (B, H)
            bag_rep = self.drop(bag_rep)
            bag_logits = self.fc(bag_rep) # (B, N)
        else:
            if bag_size == 0:
                bag_rep = []
                for i in range(len(scope)):
                    bag_rep.append(rep[scope[i][0]:scope[i][1]].mean(0))
                bag_rep = torch.stack(bag_rep, 0) # (B, H)
            else:
                batch_size = rep.size(0) // bag_size
                rep = rep.view(batch_size, bag_size, -1) # (B, bag, H)
                bag_rep = rep.mean(1) # (B, H)
            bag_logits = self.softmax(self.fc(bag_rep)) # (B, N)
        return bag_logits
