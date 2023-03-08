import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def img_to_tensor(img):
    transforms = T.Compose([
        T.Resize(226),
        T.RandomCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    img_tensor = transforms(img).unsqueeze(0)
    return img_tensor


# generate caption
def get_caps_from_image(model, img_tensor, device='cpu'):
    # generate the caption
    model.eval()
    with torch.no_grad():
        features = model.encoder(img_tensor.to(device))
        caps, alphas = model.decoder.generate_caption(features)

    return caps, alphas


def show_image(img, title=None):
    """Imshow for Tensor."""
    means = np.array([[[0.485]], [[0.456]], [[0.406]]])
    var = np.array([[[0.229]], [[0.224]],[[0.225]]])
    # unnormalize
    img = img * var + means

    img = img.numpy().transpose((1, 2, 0))

    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def save_model(model, num_epochs):
    model_state = {
        'num_epochs': num_epochs,
        'embed_size': model.decoder.embed_size,
        'vocab_size': model.decoder.vocab_size,
        'attention_dim': model.decoder.attention_dim,
        'encoder_dim': model.decoder.encoder_dim,
        'decoder_dim': model.decoder.decoder_dim,
        'state_dict': model.state_dict()
    }

    torch.save(model_state, 'attention_model_state.pth')