---
layout: page
permalink: /software/
title: Software
description: The following projects were created as part of my Ph.D.
nav: true
nav_order: 6
---

### Twitter FastText and Word2vec embeddings.

In my PhD thesis, I made a comparison between FastText and Word2vec word embeddings for PoS tagging and NER in Twitter microposts. You can find the embeddings on [Github](https://github.com/FredericGodin/TwitterEmbeddings).

### Contextual Decomposition for CNNs

As part of [my paper](https://arxiv.org/abs/1808.09551), I have implemented a decomposition technique for convolutional neural networks. I allows you to understand which patterns the neural networks is looking for to make a certain decision. In my case, I was interested in comparing character-level patterns of CNNs and BiLSTMs. The implementation can be found [here](https://github.com/FredericGodin/ContextualDecomposition-NLP).

### Twitter Word2vec model (WNUT Challenge)

As part of our ACL W-NUT 2015 shared task paper, we release a Twitter word2vec model trained on 400 million tweets, as described in detail in [this paper](https://github.com/FredericGodin/ContextualDecomposition-NLP). The code for this is currently not available but I recommend using the newer embeddings which you can find on [Github](https://github.com/FredericGodin/TwitterEmbeddings).

### Dynamic Convolutional Neural Networks

I have implemented the [paper](http://nal.co/papers/Kalchbrenner_DCNN_ACL14) “A Convolutional Neural Network for Modelling Sentences” from Kalchbrenner et al. You can find the Theano implementation [here](http://nal.co/papers/Kalchbrenner_DCNN_ACL14).
