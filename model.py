import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from numpy.random import choice
import numpy as np


class EncoderCNN(nn.Module):
    def __init__(self):
        super(EncoderCNN, self).__init__()
        resnet = torch.jit.load('model_weights/resnet50.pth').eval()
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        features = self.resnet(images)  # (batch_size,2048,7,7)
        features = features.permute(0, 2, 3, 1)  # (batch_size,7,7,2048)
        features = features.view(features.size(0), -1, features.size(-1))  # (batch_size,49,2048)
        return features


# Bahdanau Attention
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()

        self.attention_dim = attention_dim

        self.W = nn.Linear(decoder_dim, attention_dim)
        self.U = nn.Linear(encoder_dim, attention_dim)

        self.A = nn.Linear(attention_dim, 1)

    def forward(self, features, hidden_state):
        u_hs = self.U(features)  # (batch_size,num_layers,attention_dim)
        w_ah = self.W(hidden_state)  # (batch_size,attention_dim)

        combined_states = torch.tanh(u_hs + w_ah.unsqueeze(1))  # (batch_size,num_layers,attemtion_dim)

        attention_scores = self.A(combined_states)  # (batch_size,num_layers,1)
        attention_scores = attention_scores.squeeze(2)  # (batch_size,num_layers)

        alpha = F.softmax(attention_scores, dim=1)  # (batch_size,num_layers)

        attention_weights = features * alpha.unsqueeze(2)  # (batch_size,num_layers,features_dim)
        attention_weights = attention_weights.sum(dim=1)  # (batch_size,num_layers)

        return alpha, attention_weights


# Attention Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim,
                 decoder_dim, vocab, drop_prob=0.3, device='cpu'):
        super().__init__()

        # save the model param
        self.vocab_size = vocab_size
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.encoder_dim = encoder_dim
        self.embed_size = embed_size
        self.vocab = vocab
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_size + encoder_dim, decoder_dim, bias=True)
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)

        self.fcn = nn.Linear(decoder_dim, vocab_size)
        self.drop = nn.Dropout(drop_prob)
        self.sm = nn.Softmax(dim=1)

    def forward(self, features, captions):

        # vectorize the caption
        embeds = self.embedding(captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        # get the seq length to iterate
        seq_length = len(captions[0]) - 1  # Exclude the last one
        batch_size = captions.size(0)
        num_features = features.size(1)

        preds = torch.zeros(batch_size, seq_length, self.vocab_size).to(self.device)
        alphas = torch.zeros(batch_size, seq_length, num_features).to(self.device)

        for s in range(seq_length):
            alpha, context = self.attention(features, h)
            lstm_input = torch.cat((embeds[:, s], context), dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))

            output = self.fcn(self.drop(h))

            preds[:, s] = output
            alphas[:, s] = alpha

        return preds, alphas

    def generate_caption(self, features, bs=3, temperature=20, max_len=20):
        # Inference part
        # Given the image features generate the captions

        h, c = self.init_hidden_state(features)  # (batch_size, decoder_dim)

        storage = [[([1], [], 1)] * bs][0]
        end_storage = []

        for i in range(max_len):
            new_storage = []
            for indcaps, alphas, prob in storage:
                word = torch.tensor(indcaps[-1]).view(1, -1).to(self.device)
                embeds = self.embedding(word)

                alpha, context = self.attention(features, h)

                # store the apla score
                alphas.append(alpha.cpu().detach().numpy())

                lstm_input = torch.cat((embeds[:, 0], context), dim=1)
                h, c = self.lstm_cell(lstm_input, (h, c))
                output = self.sm(self.fcn(self.drop(h)) / temperature)
                output = output.view(-1).numpy()
                next_indcaps = choice(output.size, p=output, size=bs, replace=False)
                next_probs = output[next_indcaps]

                for next_prob, next_indcap in zip(next_probs, next_indcaps):
                    new_storage += [(indcaps + [int(next_indcap)], alphas, prob * next_prob)]

            new_storage = sorted(new_storage, key=lambda x: -x[2])[:bs]

            storage = []
            for indcaps, alphas, prob in new_storage:
                if self.vocab.itos[indcaps[-1]] != "<EOS>":
                    storage += [(indcaps, alphas, prob)]
                else:
                    end_storage += [(indcaps, alphas, prob)]
                    bs -= 1

            if not storage:
                break

        if not end_storage:
            end_storage = storage

        end_storage = sorted(end_storage, key=lambda x: -x[2] ** (1 / len(x[0])))
        probs = np.array([item[2] ** (1 / len(item[0])) for item in end_storage])
        indx = choice(len(end_storage), p=probs / sum(probs))
        indcaps, alphas, prob = end_storage[indx]

        if self.vocab.itos[indcaps[-1]] == '<EOS>':
            return [self.vocab.itos[idx] for idx in indcaps[1:-1]], alphas
        else:
            return [self.vocab.itos[idx] for idx in indcaps[1:]], alphas

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, attention_dim, encoder_dim,
                 decoder_dim, vocab, drop_prob=0.3, device='cpu'):
        super().__init__()
        self.encoder = EncoderCNN()
        self.decoder = DecoderRNN(
            embed_size=embed_size,
            vocab_size=vocab_size,
            attention_dim=attention_dim,
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            vocab=vocab,
            drop_prob=drop_prob,
            device=device
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs
