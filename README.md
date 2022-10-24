# IEEE Short Course 4: Introduction to Machine Learning

## Installation

### Python
This course uses Python version 3.10. It is recommended to install the Anaconda version as below:
- Install from Anaconda (recommended): Navigate to https://www.anaconda.com/ and download/install the version for your machine.

### Packages
This course makes use of some common open-source machine learning and data science tools such as Tensorflow, scikit-learn, numpy, and scipy.

1. Create new anaconda environment: conda create -n ieee_sc4_env python=3.10.4
2. Activate anaconda environment: conda activate ieee_sc4_env
3. Install requirements 

(Apple Silicon Users): 
4. conda install numpy scipy scikit-learn matplotlib seaborn tqdm notebook
5. conda install -c apple tensorflow-deps
6. python -m pip install tensorflow-macos
7. python -m pip install tensorflow-metal

(All Other Users):
4. conda install numpy scipy scikit-learn matplotlib seaborn tqdm tensorflow notebook

## Usage

This code uses Jupyter notebooks for interactive development and exploration.

1. Navigate to base directory of code
2. Start a Jupyter notebook server by typing in the command line: jupyter notebook
3. The program will tell you which link it is serving on, usually localhost:8888. Open a browser and navigate to the link.
4. Once in the browser, you will be able to navigate to each module and open up the notebooks (.ipynb files)
5. Once in a notebook, you can run a cell by clicking on the cell and pressing SHIFT+ENTER. You can also run all cells from the menu
6. If you are not familar with Jupyter notebook, take a quick read to learn about it: https://www.codecademy.com/article/how-to-use-jupyter-notebooks
