from torch import nn
import torch


class SimpleTextCNN(nn.Module):
    def __init__(self, nwords, embedding_dim, nfilters=100, convlen=(3, 4, 5),
                 dropout=0.5, nclasses=5, pretrained_embeddings=None,
                 freeze_embeddings=False):
        """
        nwords: Number of words in the vocabulary
        embedding_dim: Dimension of the word embedding vector
        nfilters: Number of output filters for the convolution layer
        convlen: Convolution sizes (A tuple)
        dropout: Dropout probability for the fully connected layer
        nclasses: Number of classes into which we're classifying
        pretrained_embeddings: Pretrained word embeddings (as a torch tensor)
        """
        super(SimpleTextCNN, self).__init__()
        
        self.nfilters = nfilters
        
        # Embedding matrix
        if pretrained_embeddings is not None:
            print('Loading pretrained embeddings')
            print('freeze_embeddings is {}'.format(freeze_embeddings))
            self.emb_matrix = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=freeze_embeddings
            ) # (nwords x EMBEDDING_DIM)
        else:
            print('Randomly initializing word embeddings')
            self.emb_matrix = nn.Embedding(nwords, embedding_dim) # (nwords x EMBEDDING_DIM)
            torch.nn.init.uniform_(self.emb_matrix.weight, -0.25, 0.25)
            torch.nn.init.zeros_(self.emb_matrix.weight[0])

        # Convolution layer
        self.convlayers = nn.ModuleList()

        for convsize in convlen:
            self.convlayers.append(nn.Conv2d(1, nfilters, (convsize, embedding_dim), padding=(convsize-1, 0))) # (N x nfilters x numblocks x 1)

        # Dropout for the final fully connected layer
        self.dropout = nn.Dropout(dropout)
        
        # Final fully connected layer
        self.fc1 = nn.Linear(nfilters*len(convlen), nclasses)

    def forward(self, x):
        """
        Define the forward pass
        1. Select word vectors from the embedding matrix
        2. Run a convolution pass on the tensor
        3. Max pool over the tensor
        4. Apply Relu on the result
        """
        embeddings = self.emb_matrix(x)
        # Ensure that embedding for "0" (padding) goes to zero.
        embeddings = embeddings * torch.sign(x).unsqueeze(-1).float()
        embeddings = embeddings.unsqueeze(1)

        convmaxpools = []

        for conv in self.convlayers:
            convol = conv(embeddings)
            maxpool_func = nn.MaxPool1d(convol.shape[2])
            maxpool = maxpool_func(convol.squeeze(3))
            maxpool = maxpool.squeeze(-1)
            convmaxpools.append(maxpool)
        
        maxpool_result = torch.cat(convmaxpools, 1)
        
        relu = nn.functional.relu(maxpool_result)
        relu = self.dropout(relu)
        logits = self.fc1(relu)
        
        return logits


class SimpleBOW(nn.Module):
    def __init__(self, nwords, embedding_dim, nclasses=5,
                 pretrained_embeddings=None):
        """
        nwords: Number of words in the vocabulary
        embedding_dim: Dimension of the word embedding vector
        nclasses: Number of classes into which we're classifying
        pretrained_embeddings: Pretrained word embeddings (as a torch tensor)
        """
        super(SimpleBOW, self).__init__()

        # Embedding matrix
        if pretrained_embeddings is not None:
            print('Loading pretrained embeddings')
            self.emb_matrix = nn.Embedding.from_pretrained(
                pretrained_embeddings,
                freeze=False
            ) # (nwords x EMBEDDING_DIM)
        else:
            print('Randomly initializing word embeddings')
            self.emb_matrix = nn.Embedding(nwords, embedding_dim) # (nwords x EMBEDDING_DIM)
            torch.nn.init.uniform_(self.emb_matrix.weight, -0.25, 0.25)
            torch.nn.init.zeros_(self.emb_matrix.weight[0])

        self.linear = nn.Linear(embedding_dim, nclasses)

    def forward(self, x):
        """
        A simple CBOW model. Does the following -
        1. Lookup embeddings corresponding to words in the sentence.
        2. Average these embeddings to create an embedding dimensional
            representation.
        3. Pass this through a fully connected layer to predict the class.
        """
        embeddings = self.emb_matrix(x)

        # Ensure that embedding for "0" (padding) goes to zero.
        embeddings = embeddings * torch.sign(x).unsqueeze(-1).float()
        emb_mean = torch.mean(embeddings, dim=1).squeeze(1)
        emb_relu = nn.functional.relu(emb_mean)

        logit = self.linear(emb_relu)
        return logit
