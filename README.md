# Image-Captoning-Model
A model for describing what is depicted in the photos

------------------------------------

Image Captioning model - это модель, которая по фотографии выдает её описание.

Делал я это для себя  и целью было ознакомиться и реализовать модель с задачей описания изображения.
Для использования модели я выбрал библиотеку `Stremlit`, так как он прост и эффективен. Так же его можно выгрузить в `StreamlitCloud` бесплатно, что большой плюс.

Чтобы запустить все локально нужно ввести в командную строку или терминал в `PyCharm` команду

`streamlit run web.py`.

Датасет
------------------------------------
В качестве датасета был взят [Flickr 8k](https://www.kaggle.com/datasets/adityajn105/flickr8k/versions/1). Изначально хотелось взять датасет, содержащий описания на русском, но такового не нашлось. 

Можно было взять и [Flickr 30k](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset), но она показалась слишком большой при маленькой модели. 

Планируется дальше обучить уже другую более большую сеть уже на этом датасете.

Модель
------------------------------------
В качестве модели был взят `Encder` `ResNet-50`, а `Decoder`  обычная рекурентная сеть с `LSTM`. 

В папке `model_weights` лежат веса данной модели, которые подгружаются при инициализации веб-демо.

Результаты
------------------------------------

<p align='center'>
  <img src='photos/park.jpg' height='512' width='512'/>
  <img src='photos/sunglasses.jpg' height='512' width='400'/>
</p>

Результаты, коненчо, неочень, нужно побольше модель и подольше её пообучать. 

Источники
------------------------------------
Датасет: <br />
https://github.com/ari-dasci/OD-WeaponDetection <br />

Streamlit:<br />
https://streamlit.io/ <br />
https://habr.com/ru/post/664076/ <br />
https://medium.com/nuances-of-programming/как-развернуть-веб-приложение-streamlit-в-сети-три-простых-способа-3fe4bdbbd0a9  <br />

Tutorial for Image Captioning: <br />
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning <br />
