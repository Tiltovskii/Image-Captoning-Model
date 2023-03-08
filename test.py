from dataset import Vocabulary
from configure import link_to_vocab

vocab = Vocabulary(freq_threshold=1)
vocab.download(link_to_vocab)

print(vocab.stoi)

print(vocab.itos)

