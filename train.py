from tqdm import tqdm

import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import EncoderDecoder
from dataset import FlickrDataset, CapsCollate
from utils import save_model, show_image


def train(data_loader, model, optimizer, criterion, num_epochs, print_every, device='cpu'):
    for epoch in tqdm(range(1, num_epochs + 1)):
        for idx, (image, captions) in enumerate(iter(data_loader)):
            image, captions = image.to(device), captions.to(device)

            # Zero the gradients.
            optimizer.zero_grad()

            # Feed forward
            outputs, attentions = model(image, captions)

            # Calculate the batch loss.
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

            # Backward pass.
            loss.backward()

            # Update the parameters in the optimizer.
            optimizer.step()

            if (idx + 1) % print_every == 0:
                print("Epoch: {} loss: {:.5f}".format(epoch, loss.item()))

                # generate the caption
                model.eval()
                with torch.no_grad():
                    dataiter = iter(data_loader)
                    img, _ = next(dataiter)
                    features = model.encoder(img[0:1].to(device))
                    caps, alphas = model.decoder.generate_caption(features)
                    caption = ' '.join(caps)
                    show_image(img[0], title=caption)

                model.train()

        # save the latest model
        save_model(model, epoch)


if __name__ == '__main__':
    # setting the constants
    data_location = "your folder"
    BATCH_SIZE = 256
    NUM_WORKER = 2

    # defining the transform to be applied
    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # testing the dataset class
    dataset = FlickrDataset(
        root_dir=data_location + "/Images",
        caption_file=data_location + "/captions.txt",
        transform=transforms
    )

    # writing the dataloader
    # token to represent the padding
    pad_idx = dataset.vocab.stoi["<PAD>"]

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
        collate_fn=CapsCollate(pad_idx=pad_idx, batch_first=True)
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    embed_size = 300
    vocab_size = len(dataset.vocab)
    attention_dim = 256
    encoder_dim = 2048
    decoder_dim = 512
    learning_rate = 3e-4

    # init model
    model = EncoderDecoder(
        embed_size=300,
        vocab_size=len(dataset.vocab),
        attention_dim=256,
        encoder_dim=2048,
        decoder_dim=512,
        vocab=dataset.vocab,
        device=device
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 25
    print_every = 100
    train(data_loader, model, optimizer, criterion, num_epochs, print_every, device=device)


