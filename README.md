# DivideMix: Learning with Noisy Labels as Semi-supervised Learning
PyTorch Code for the following paper:\
<b>Title</b>: <i>DivideMix: Learning with Noisy Labels as Semi-supervised Learning</i> <a href="https://openreview.net/pdf?id=SkxhErJIvB">[pdf]</a>\
<b>Authors</b>:Junnan Li, Steven C.H. Hoi, Richard Socher\
<b>Institute</b>: Salesforce Research Asia


<b>Abstract</b>\
Deep neural networks are known to be annotation-hungry. Numerous efforts have been devoted to reduce the annotation cost when learning with deep networks. Two prominent directions include learning with noisy labels and semi-supervised learning by exploiting unlabeled data. In this work, we propose DivideMix, a novel framework for learning with noisy labels by leveraging semi-supervised learning techniques. In particular, DivideMix models the per-sample loss distribution with a mixture model to dynamically divide the training data into a labeled set with clean samples and an unlabeled set with noisy samples, and trains the model on both the labeled and unlabeled data in a semi-supervised manner. To avoid confirmation bias, we simultaneously train two diverged networks where each network uses the dataset division from the other network. During the semi-supervised training phase, we improve the MixMatch strategy by performing label co-refinement and label co-guessing on labeled and unlabeled samples, respectively. Experiments on multiple benchmark datasets demonstrate substantial improvements over state-of-the-art methods.


<b>Illustration</b>\
<img src="./img/framework.png">

<b>Experiments</b>\
First, please create a folder named <i>checkpoint</i>to store the results.\
Next, run Train_xx.py --data_path <i>path-to-your-data</i>

<b>Cite DivideMix</b>\
If you find the code useful in your research, please consider citing our paper: