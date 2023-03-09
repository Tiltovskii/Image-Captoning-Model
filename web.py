import io
import streamlit as st
from model import EncoderDecoder
from PIL import Image
import torch
import gdown
import numpy as np
import matplotlib.pyplot as plt

from dataset import Vocabulary
from configure import link_to_vocab, url
from utils import get_caps_from_image, img_to_tensor


def load_image():
    """Создание формы для загрузки изображения"""
    # Форма для загрузки изображения средствами Streamlit
    uploaded_file = st.file_uploader(
        label='Выберите изображение для описания')
    if uploaded_file is not None:
        # Получение загруженного изображения
        image_data = uploaded_file.getvalue()
        # Показ загруженного изображения на Web-странице средствами Streamlit
        st.image(image_data)
        # Возврат изображения в формате PIL
        return Image.open(io.BytesIO(image_data))
    else:
        return None


@st.cache(allow_output_mutation=True)
def load_model():
    with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
        vocab = Vocabulary(freq_threshold=1)
        vocab.download(link_to_vocab)

        model = EncoderDecoder(
            embed_size=300,
            vocab_size=len(vocab),
            attention_dim=256,
            encoder_dim=2048,
            decoder_dim=512,
            vocab=vocab
        )

        try:
            model.load_state_dict(torch.load('model_weights/image_captioning_model.pth',
                                             map_location=torch.device('cpu'))['state_dict'])
        except Exception as exc:
            print(exc)
            gdown.download_folder(url, quiet=True, use_cookies=False)
            model.load_state_dict(torch.load('model_weights/image_captioning_model.pth')['state_dict'])

    return model


def write_caps(caps):
    caption = ' '.join(caps)
    st.write(caption)


def plot_attention(img, result, attention_plot):
    img = img.squeeze(0)
    means = np.array([[[0.485]], [[0.456]], [[0.406]]])
    var = np.array([[[0.229]], [[0.224]], [[0.225]]])
    # unnormalize
    img = img * var + means
    img = img.numpy().transpose((1, 2, 0))
    temp_image = img
    fig = plt.figure(figsize=(15, 15))

    len_result = len(result)
    for l in range(len_result):
        temp_att = attention_plot[l].reshape(7, 7)

        ax = fig.add_subplot(len_result // 2, len_result // 2, l + 1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.7, extent=img.get_extent())

    plt.tight_layout()
    st.pyplot(fig)


model = load_model()
# Выводим заголовок страницы средствами Streamlit
st.title('Описание изображений')
# Вызываем функцию создания формы загрузки изображения
img = load_image()

result = st.button('Описать изображение')
# Если кнопка нажата, то запускаем распознавание изображения
if result and img:
    img_tensor = img_to_tensor(img)
    caps, alphas = get_caps_from_image(model, img_tensor, bs=1, temperature=1.2, max_len=30)
    write_caps(caps)
    with st.spinner("Рисуем attention map! :sunglasses:"):
        plot_attention(img_tensor, caps, alphas)
