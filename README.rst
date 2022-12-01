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
    
Use-case example is German-to-English machine translation.

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

Data
----

The model is trained on 30k English-German translation dataset: https://pytorch.org/text/stable/datasets.html#multi30k 

=====
Reproduce the results
=====

1. Get model weights
-----
The pre-trained weights are stored on my Drive. You can use the link below to add the shortcut (no need to download)

- Add shortcut to Google drive: https://drive.google.com/file/d/1fQLCFoj2-RmS1M-LzyaSlUYVOS8LSrE1/view?usp=sharing


2. Load IPython
-----
Prepare a colab session

- Go to https://colab.research.google.com/ 
- select 'GitHub' and past https://github.com/nikitakapitan/transformers
- choose [DEMO]Predict.ipynb

3. Run the notebook
----
Simply execute the cells. It will:

- install and import required packages
- mount your Google Drive and copy **model weights** to colab session
 - at this point you should have **multi30k_model_final.pt** file in your Google Drive from step 1.
- **check_outputs** function prints the results from the table above.

Note: you won't see the exact same sentences due to random batch. But the quality of translation will remain the same.




