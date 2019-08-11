# NNA : Perceptron Model

## Perceptron Class

    `perceptron.py` : This script has the class definations.
    Look into the code for detailed comments.

## Data Generator

    `data_generator.py` : This script deals with generating uniformly at random normally distributed data make sure to set the parameters right, and follow the comments.

## 2-Class Classification

    `2-class-classification.py` : This script has the code for 2 class classification.

    The parameters that can be tweaked are ,

    `DIMENSION_OF_X` : Total number of features/independent varaibles for any given sample.
    `TOTAL_SAMPLES` : Total number of sample.
    `list_of_means` : mean of data for each of the classes, used by the data generator.
    `list_of_std` : standard deviations of the data for each of the classes, used by the data generator.
    `epochs` : training epochs.
    `learning_rate` : learning rate for the perceptron.

    Additional : If number of `dimensions == 2` then graph would be plotted.


## Reference

    1. [Perceptron Model](https://medium.com/@thomascountz19-line-line-by-line-python-perceptron-b6f113b161f3)
    2. [Adding a column to matrix of numpy](https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array)
    3. [For Activation Function, if value > 0 then 1 else 0](https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1)