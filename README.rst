.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/transformers.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/transformers
    .. image:: https://readthedocs.org/projects/transformers/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://transformers.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/transformers/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/transformers
    .. image:: https://img.shields.io/pypi/v/transformers.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/transformers/
    .. image:: https://img.shields.io/conda/vn/conda-forge/transformers.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/transformers
    .. image:: https://pepy.tech/badge/transformers/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/transformers
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/transformers

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

============
Transformers
============
This is my replication from scratch of the article Attentino is all you need: https://arxiv.org/abs/1706.03762



Data 
----
Use-case example is German-to-English machine translation.
The model is trained on 30k English-German translation dataset: https://pytorch.org/text/stable/datasets.html#multi30k 


Results
-------
.. list-table:: 
   :widths: 25 25 50
   :header-rows: 1

   * - Source text - German
     - Ground truth - English
     - Model Output
   * - Drei Männer auf Pferden während eines Rennens 
     - Three men on horses during a race
     - Three men on horseback during a race
   * - Ein Kind in einem orangen Shirt springt von Heuballen herunter , während andere Kinder zusehen
     - A child in an orange shirt jumps off bales of hay while other children watch
     - A child in an orange shirt is jumping down a dirty street while other children watch him
   * - Zwei Männer in Shorts arbeiten an einem blauen Fahrrad 
     - Two men wearing shorts are working on a blue bike
     - Two men in shorts are working on a blue bicycle 
   * - Kinder einer Schulklasse sprechen miteinander und lernen
     - Kids conversing and learning in class
     - A group of kids are having a conversation and making a conversation 

Architecture
------------
This project replicates the original architecture of Transformers which is basically Encoder-Deconder model with a lot of attentions.

   
.. image:: https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png
    :width: 300



=====
Reproduce the results
=====

1. Train your own model from scratch.
-----
- Go to https://colab.research.google.com/ 
- select 'GitHub' and past https://github.com/nikitakapitan/transformers
- choose [DEMO]Train.ipynb

By executing all cells you will connect to your google drive (a place where the model weights will be saved).

Then you will eventually launch the training (about 15 minutes on colab's GPU).

The final model weights are stored in **multi30k_model_final.pt** file.

The last command (!cp multi30...) will copy the weigts from colab VM to your Google Drive.

Now you have trained weights on you Google Drive 🤗


2. Laucn translation on validation dataset.
-----

- Go to https://colab.research.google.com/ 
- select 'GitHub' and past https://github.com/nikitakapitan/transformers
- choose [DEMO]Predict.ipynb

By executing all cells you will connect to your google drive (a place where the model will search for its weights **multi30k_model_final.pt**)

Then you will create a validation data set containing GER-ENG pairs unseen my the model during training.

Then you will eventually launch the prediction (i.e. translation).


Note: Due to random batch, you won't see the exact same sentences as above, but the quality should remain the same.




