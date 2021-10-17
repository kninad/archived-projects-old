# README

There are two Python scripts `blr.py` and `misc.py`. `blr.py` contains the 
functions required by the homework handout. `misc.py` has some utility functions
to produce the requried plots. All the functions in `blr.py` have a docstring 
for ease in understanding their purpose. 

All the code was written assuming a Python 3 environment.


## Reproducing the plots

To reproduce the submitted plots, just run `misc.py` and all the plots
will be written to a folder `out` which is assumed to be present at the same 
tree level as the `code` directory.

Running `misc.py` also writes out a pickle file `samples_s100.pk` which contains
a dictionary storing the 100 theta samples, each for M = 10, 30 and 50.


