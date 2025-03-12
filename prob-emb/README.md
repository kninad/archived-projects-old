# prob-emb
Code repository for cs696 project on embeddings.

## Probabilistic embeddings on taxonomies

Learning representations of symbolic data such as texts and graphs is an integral part of 
machine learning practice with broad applications in information extraction.  
However, many embedding methods in Euclidean space fail to account for the hierarchical 
structure inherent in many symbolic datasets such as with ontologies and knowledge graphs.
Our work seeks to explore techniques such as Probabilistic Embeddings with Box Lattice 
measures[1], to induce taxonomies and relational graphs.  The internship project will 
expand upon current techniques with an opportunity to explore and develop other 
non-euclidean techniques for hierarchical embeddings [2, 3].

[1] Probabilistic Embedding of Knowledge Graphs with Box Lattice. (ACL 2018)

[4] Smoothing The Geometry of Probabilistic Box Embeddings (ICLR 2019)

[2] Poincaré Embeddings for Learning Hierarchical Representations. (NIPS 2017)

[3] Learning Continous Hierarchies in the Lorentz Model of Hyperbolic Geomentry. (ICML 2018)


## Requirements

Python 3.6, PyTorch (hyperb-emb) and Tensorflow (prob-box-emb) and others (numpy,pandas)

## Things to follow
- Use 4 spaces for a TAB. Convert all TABS to spaces
- Dont upload large data files to github. Put them up on google drive. 
- Add un-necessary files to the .gitignore file so that these files are not tracked
  (eg. .pyc, data files, .ipynb_checkpoints/ etc)
- Write comments and use docstrings for function for better documentation.
  Use the following for reference: 
  [https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard](numpy-docstring-guide)
- Use the google drive folder for all project materials. data, todolist, documents, notes etc.
