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


.. architecture:: = {
        'src_vocab_len' : 8315, # GERMAN vocab size
        'tgt_vocab_len' : 6384, # ENGLISH vocab size
        'N' : 6,                # nb of loops in Transformer
        'd_model' : 512,        # Model size aka Input size aka Embedding size
        'd_ff' : 2048,          # nb of neurons in Linear layer
        'h' : 8,                # nb of attention heads
        'p_dropout' : 0.1       # dropout probability (for training)
    }


=====
Reproduce the results
=====

To reproduce the results you will need no more than execute the cells in google colab 🤗

Basically, we will need only NumPy, PyTorch and some side libs, no more.

- We will not import any models  : instead we will build the Transformer from scratch using torch.
- We will not import any weights : instead we will train the model from scratch on row data.

So, let's rock and roll:

1. Build and rain the model from scratch.
-----
- Go to https://colab.research.google.com/ 
- select 'GitHub' and past https://github.com/nikitakapitan/transformers
- choose [DEMO]Train.ipynb

By executing all cells you will connect to your google drive (a place where the model weights will be saved).

First, you will build the model using **make_model** function and settings dictionary **achitecture**

Then you will eventually launch the training on GER-ENG pairs dataset (about 15 minutes on colab's GPU).

The final model weights will be stored in **multi30k_model_final.pt** file.

The last command (!cp multi30...) will copy the weigts from current colab session to your Google Drive.

Now you have trained weights on you Google Drive 🤗


2. Laucn translation on validation dataset.
-----

- Go to https://colab.research.google.com/ 
- select 'GitHub' and past https://github.com/nikitakapitan/transformers
- choose [DEMO]Translate.ipynb

By executing all cells you will connect to your google drive (a place where the model will search for weights **multi30k_model_final.pt**)
​
134
Note: Due to random batch, you won't see the exact same sentences as above, but the quality should remain the same.
Then please define your own phrase inside **YOUR_GERMAN_SENTENCE**

Finally, you will see the English translation as the output. 


Note : 
----
If you don't want to input your own german phrase, during step 2 select [DEMO]Predict.ipynb.

This will translate some samples from validation dataset (unseen during the training).
