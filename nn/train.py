import argparse
from torch import nn
import torch
import utils
import model
import pickle
import numpy as np
from tqdm import tqdm
import logging
import os


parser = argparse.ArgumentParser(description='Train sentence classifier.')
parser.add_argument('--init', type=str, required=True)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--valid', type=str, required=True)
parser.add_argument('--test', type=str, required=True)

parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--pretrained_file', type=str, required=False)
parser.add_argument('--freeze_embeddings', dest='freeze_embeddings', action='store_true')
parser.add_argument('--optimizer', type=str, default='Adam', required=False)
parser.add_argument('--to_lower', dest='to_lower', action='store_true')
parser.add_argument('--cbow', dest='cbow', action='store_true')

parser.add_argument('--output_path', type=str, required=True)

CPU = 'cpu'
GPU = 'cuda:0'

RANDOM = 'random'

if torch.cuda.is_available():
    device = GPU
else:
    device = CPU

if __name__ == '__main__':
    args = parser.parse_args()
    init = args.init

    if init != RANDOM and args.pretrained_file is None:
        parser.error('Pretrained file must be specified if training in non random mode')

    print ('Training on {}'.format(device))

    # Load data
    corpus = utils.load_data(args.train,
                             args.valid,
                             args.test,
                             to_lower=args.to_lower)

    # Number of training samples
    print('Number of training samples {}'.format(len(corpus.training_input)))
    print('Number of validation samples {}'.format(len(corpus.dev_input)))

    print('Number of words {}'.format(len(corpus.w2i)))
    print('Number of labels {}'.format(len(corpus.t2i)))

    print('Labels {}'.format(corpus.t2i.keys()))

    print('to_lower is {}'.format(args.to_lower))


    training_inputs = [
    [
        torch.tensor(elem, device=device, dtype=torch.long)
            for elem in corpus.training_input[ind:ind+args.batch_size]
    ] for ind in range(0, len(corpus.training_input), args.batch_size)]
    training_inputs = [nn.utils.rnn.pad_sequence(sequence).t()
                        for sequence in training_inputs]

    training_labels = [corpus.training_labels[ind:ind+args.batch_size] 
                        for ind in range(0, len(corpus.training_labels), args.batch_size)]
    training_labels = [torch.tensor(sequence, device=device, dtype=torch.long)
                        for sequence in training_labels]

    training_batches = [torch.cat([inp, label.view(-1,1)], dim=1)
                        for inp, label in zip(training_inputs, training_labels)]

    dev_inputs = [torch.tensor(elem, device=device, dtype=torch.long).unsqueeze(0)
                        for elem in corpus.dev_input]
    dev_labels = [torch.tensor(elem, device=device, dtype=torch.long).unsqueeze(0)
                        for elem in corpus.dev_labels]

    embedding_matrix = None
    if args.pretrained_file:
        embedding_matrix = torch.tensor(
            utils.word2vec_populate_pretrained_embeddings(args.pretrained_file,
                                                        corpus, 300),
            device=device, dtype=torch.float
        )
    
    model_base_path = args.output_path
    os.makedirs(model_base_path, exist_ok=True)

    # Training phase

    if not args.cbow:
        model = model.SimpleTextCNN(nwords=len(corpus.w2i), embedding_dim=300,
                                nfilters=100, nclasses=len(corpus.t2i),
                                pretrained_embeddings=embedding_matrix,
                                freeze_embeddings=args.freeze_embeddings)
        print('Training CNN')
    else:
        model = model.SimpleBOW(len(corpus.w2i), 300,
                                len(corpus.t2i), pretrained_embeddings=embedding_matrix)
        print('Training CBOW')

    if device == GPU:
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    print ('Training with {} optimizer'.format(args.optimizer))
    criterion = nn.CrossEntropyLoss()
    
    min_val_accuracy = 0
    
    for EPOCH in range(10):
        model.train()
        for training_batch in tqdm(training_batches):

            # Permute the input
            perm = torch.randperm(len(training_batch))
            training_batch = training_batch[perm]

            input_batch = training_batch.narrow(1, 0, training_batch.shape[1]-1)
            labels = training_batch[:, -1]

            logit = model(input_batch)
            loss = criterion(logit, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print('{} training epoch done'.format(EPOCH))

        model.eval()

        correct_count = 0
        total_count = 0
        val_loss = 0

        for input_batch, labels in zip(dev_inputs, dev_labels):
            logit = model(input_batch)
            val_loss += criterion(logit, labels)

            correct_count += torch.sum(torch.eq(logit.argmax(dim=1), labels))
            total_count += len(input_batch)

        accuracy = 100*(correct_count.item())/total_count
        print('Correct predictions {}/{} = {}. Loss value {}'.format(
            correct_count, total_count, accuracy, val_loss)
        )

        if accuracy > min_val_accuracy:
            torch.save(model.state_dict(), os.path.join(
                model_base_path, './torch_{}'.format(EPOCH))
            )
            print('Saving model for epoch {}'.format(EPOCH))
            min_val_accuracy = accuracy
