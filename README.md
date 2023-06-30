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
@inproceedings{
bauvin2023sample,
title={Sample Boosting Algorithm (Sam{BA}) - An Interpretable Greedy Ensemble Classifier Based On Local Expertise For Fat Data},
author={Baptiste Bauvin and C{\'e}cile Capponi and Florence Clerc and Pascal Germain and Sokol Ko{\c{c}}o and Jacques Corbeil},
booktitle={The 39th Conference on Uncertainty in Artificial Intelligence},
year={2023},
url={https://openreview.net/forum?id=0k_DN90uWF}
}
```

Author
-------

* **Baptiste BAUVIN**
