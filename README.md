# SamBA : Sample Boosting Algorithm

This package provides Python code for SamBA, a boosting-based classifier that
uses similarity to learn a combination of local experts. For in-depth
information, see B.Bauvin et al, UAI 2023.

It heavily relies on scikit-learn, for usage examples, see the demo directory in
which the code used for the paper's figures is provided.


## Installing

Once you cloned the project, run, with python3: 

```
    cd path/to/neighborhood_classifier/
    pip install -e .
```

## Reproducing the results

To reproduce the results, we provide the `demo` directory, in which each 
experiment is coded. They are straightforward, except for the performance study
 that heavily relies on [SuMMIT](https://github.com/multi-learn/summit). 
 We are aware that the tool is too complex for the simple experiments provided 
 in the paper. As a consequence, we provided a minimal version of summit in the 
 `Performance` directory.
To reproduce the results of the paper run: 
 
```
    cd path/to/neighborhood_classifier/demo/summit_minimal_version/
    pip install -e .
    cd summit/
    python execute --config_path ../../config_summit_metagenome.yml
```

Such an execution might be long and demanding. As a consequence, we provide a 
pre-computed result directory `Performance/results/metagenome/pre_computed/`, 
that is used in the `Performance/preformance_study.py` script.

## Getting the data

The **metagenome** dataset is provided in the `Performance/dataset/` directory 
as an HDF5 file containing all the data types of the paper.

In addition, we provide a `csv` version for each data type, and one for the labels. 

## Cite the paper

```

@InProceedings{pmlr-v216-bauvin23a,
  title = 	 {Sample {B}oosting {A}lgorithm ({SamBA}) - An interpretable greedy ensemble classifier based on local expertise for fat data},
  author =       {Bauvin, Baptiste and Capponi, C\'{e}cile and Clerc, Florence and Germain, Pascal and Ko\c{c}o, Sokol and Corbeil, Jacques},
  booktitle = 	 {Proceedings of the Thirty-Ninth Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {130--140},
  year = 	 {2023},
  editor = 	 {Evans, Robin J. and Shpitser, Ilya},
  volume = 	 {216},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {31 Jul--04 Aug},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v216/bauvin23a/bauvin23a.pdf},
  url = 	 {https://proceedings.mlr.press/v216/bauvin23a.html},
  abstract = 	 {Ensemble methods are a very diverse family of algorithms with a wide range of applications. One of the most commonly used is boosting, with the prominent Adaboost. Adaboost relies on greedily learning base classifiers that rectify the error from previous iterations. Then, it combines them through a weighted majority vote, based on their quality on the entire learning set. In this paper, we propose a supervised binary classification framework that propagates the local knowledge acquired during the boosting iterations to the prediction function. Based on this general framework, we introduce SamBA, an interpretable greedy ensemble method designed for fat datasets, with a large number of dimensions and a small number of samples. SamBA learns local classifiers and combines them, using a similarity function, to optimize its efficiency in data extraction. We provide a theoretical analysis of SamBA, yielding convergence and generalization guarantees. In addition, we highlight SamBA’s empirical behavior in an extensive experimental analysis on both real biological and generated datasets, comparing it to state-of-the-art ensemble methods and similarity-based approaches.}
}
```

Author
-------

* **Baptiste BAUVIN**
