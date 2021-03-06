# NNA : Perceptron Model

## Perceptron Class

`perceptron.py` : This script has the class definations.
Look into the code for detailed comments.

## Data Generator

`data_generator.py` : This script deals with generating uniformly at random normally distributed data make sure to set the parameters right, and follow the comments.

## 2-Class Classification

`2-class-classification.py` : This script has the code for 2 class classification.

This script can handle multi dimnsional data.

The parameters that can be tweaked are ,

1. `DIMENSION_OF_X` : Total number of features/independent varaibles for any given sample.
2. `TOTAL_SAMPLES` : Total number of sample.
3. `list_of_means` : mean of data for each of the classes, used by the data generator.
4. `list_of_std` : standard deviations of the data for each of the classes, used by the data generator.
5. `epochs` : training epochs.
6. `learning_rate` : learning rate for the perceptron.
7. `ANIMATE`: Plots the training process

Additional : If number of `dimensions == 2` then graph would be plotted.

## 3-Class Classification

`3-class-classification.py` : This script handles the 3 class classification problem using only 2 perceptron.

All the parameters/knobs are same as 2-class classification except `ANIMATE`.

## Reference

1. [Perceptron Model](https://medium.com/@thomascountz19-line-line-by-line-python-perceptron-b6f113b161f3)
2. [Adding a column to matrix of numpy](https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array)
3. [For Activation Function, if value > 0 then 1 else 0](https://stackoverflow.com/questions/45648668/convert-numpy-array-to-0-or-1)
4. [Plotting line, and scatter points](https://stackoverflow.com/questions/12487060/matplotlib-color-according-to-class-labels)
5. [Dynamic Plot updates](https://stackoverflow.com/questions/4098131/how-to-update-a-plot-in-matplotlib/4098938#4098938)