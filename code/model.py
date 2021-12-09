import torch
import torch.nn as nn

class FastAccurateParserModel(nn.Module):

    """
    PyTorch implementation of model from A Fast and Accurate Dependency Parser using Neural Networks.
    """

    def __init__(self, vocab_size, e_dim, num_feats, h_dim, num_labels, dropout=0.0, embeddings=None):
        super(FastAccurateParserModel, self).__init__()
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.loss_train = nn.CrossEntropyLoss()
        self.dropout_prob = dropout
        
        if embeddings == None:
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=e_dim)
        else:
            # if weights provided, use those
            self.embeddings = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=False)

        # linear layer: xW + b1
        input_dim = e_dim * num_feats
        self.linear_layer = nn.Linear(input_dim, h_dim)

        # project to label space: hU + b2
        self.label_layer = nn.Linear(h_dim, num_labels)

        # dropout
        self.dropout = nn.Dropout(p=self.dropout_prob)

    def forward(self, x):
        #print(f"x in forward: {x.size()}")
        x = self.embeddings(x)
        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        #print(f"after embedding expansion: {x.size()}")
        # linear layer
        #print(f"linear layer shape: {self.linear_layer.weight.size()}")
        h = self.linear_layer(x)
        # cubic nonlinearity
        h = h * h * h
        # project to tag space
        p = self.label_layer(h)

        return p

    def loss(self, y_hat, y):
        return self.loss_train(y_hat, y)

