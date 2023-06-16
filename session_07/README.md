# Objective

The objective of the assignment was to train a model to achieve the following with an MNIST dataset:

- More than 99.40% testing accuracy at multiple epochs in the end
- Less than 8,000 parameters
- Less than or equal to 15 Epochs
- Show Receptive Field calculations
- Use at least 3 iterations

## Note

My receptive field calculations are in a table in a text block in the individual `.ipynb` files.

<hr>

# Iteration 1

This was just a setup of various helpers for loading data, training, testing etc.
The model was quite heavy.

### Target:

- Get the set-up right
- Set Transforms
- Set Data Loader
- Set Basic Working Code
- Set Basic Training & Test Loop

### Results:

- Parameters = 6,379,786
- Best training accuracy = 99.93%
- Best testing accuracy = 99.31%

### Analysis:

- Extremely Heavy Model for such a problem
- Model is over-fitting
- This was just to setup the structure of the code and we'll improve upon it in later iterations

Solution Notebook: `session_07_iteration_01.ipynb`
<br>

<hr>

# Iteration 2

## Target:

- We want to get the skeleton structure right so that we can add minimum changes to future iterations at a time

## Results:

- Parameters = 194,884
- Best training accuracy = 99.28%
- Best testing accuracy = 98.92%

## Analysis:

- Model is large, our target parameters = 8,000
- Model is over-fitting

Solution Notebook: `session_07_iteration_02.ipynb`
<br>

<hr>

# Iteration 3

## Target:

- To make the model lighter i.e. less number of parameters

## Results:

- Parameters = 10,790
- Best training accuracy = 98.56%
- Best testing accuracy = 98.39%

## Analysis:

- Model is still larger than our target, but it's still a good model
- Model is not over-fitting as evident by the delta in training and testing accuracy. If training accuracy is improved, I can see the model achieving the target testing accuracy

Solution Notebook: `session_07_iteration_03.ipynb`
<br>

<hr>

# Iteration 4

## Target:

- To make the model even lighter

## Results:

- Parameters = 9,990
- Best training accuracy = 99.03%
- Best testing accuracy = 98.71%

## Analysis:

- Model has not achieved its target size but it's lighter
- Model is not over-fitting as evident by the delta in training and testing accuracy. If training accuracy is improved, I can see the model achieving the target testing accuracy

Solution Notebook: `session_07_iteration_04.ipynb`
<br>

<hr>

# Iteration 5

## Target:

- Add batch-normalization to increase model efficiency.

## Results:

- Parameters = 10,154
- Best training accuracy = 99.82%
- Best testing accuracy = 99.20%

## Analysis:
- Model has increased parameters with batch normalization
- Model is overfitting as evident by accuracies from epoch 10 onwards, the delta is increasing

Solution Notebook: `session_07_iteration_05.ipynb`
<br>

<hr>

# Iteration 6

## Target:

- Add regularization (dropout) to make learning harder in hopes of increasing testing accuracy

## Results:

- Parameters = 10,154
- Best training accuracy = 99.35%
- Best testing accuracy = 99.30%

## Analysis:

- Model still hasn't achieved target size. Parameters were unaffected (as expected) after adding dropout
- Model is not over-fitting, but I could not achieve the target testing accuracy

Solution Notebook: `session_07_iteration_06.ipynb`
<br>

<hr>

# Iteration 7

## Target:

- Add GAP and remove the last convolution with big kernel.
- Achieve target size of < 8,000 parameters

## Results:

- Parameters = 5,254
- Best training accuracy = 98.56%
- Best testing accuracy = 98.98%

## Analysis:

- Model has achieved target size
- Model is not over-fitting, but the accuracy has dropped in comparison to the previous iteration, which is expected as the number of parameters has also dropped significantly.
- In the next iteration, might need to increase the number of parameters to fairly compare the accuracies.

Solution Notebook: `session_07_iteration_07.ipynb`
<br>

<hr>

# Iteration 8

## Target:

- Add more parameters to make this model comparable to a similarly sized model (before GAP)
- Max pooling after RF = 5 as it is MNIST data where features start to form at that field

## Results:

- Parameters = 13,808
- Best training accuracy = 99.37%
- Best testing accuracy = 99.46%

## Analysis:

- Model exceeded target size
- Model is not over-fitting.
- We reached our target testing accuracy

  - 99.42% at Epoch 12
  - 99.46% at Epoch 13

- Our target is not achieved consistently (during later epochs)
- I will play with the training data now to make the training harder

Solution Notebook: `session_07_iteration_08.ipynb`
<br>

<hr>

# Iteration 9

## Target:

- Added a random rotation to our training data to make the training even harder

## Results:

- Parameters = 13,808
- Best training accuracy = 99.26%
- Best testing accuracy = 99.50%

## Analysis:

- Model exceeded target size
- Model is under-fitting as expected.
- We reached our target testing accuracy

  - 99.50% at Epoch 10
  - 99.49% at Epoch 14

- Overall testing accuracy is up and we achieved our target during the last epoch. And accuracy overall is high during the later epochs.
- Since accuracy is going up and down, I'll play with learning rate in the next iteration

Solution Notebook: `session_07_iteration_09.ipynb`
<br>

<hr>

# Iteration 10

## Target:

- Added StepLR Scheduler to change learning rate

## Results:

- Parameters = 13,808
- Best training accuracy = 99.24%
- Best testing accuracy = 99.44%

## Analysis:

- Model exceeded target size
- Model is under-fitting as expected.
- We reached our target testing accuracy faster and more consistently
  - 99.42% at Epoch 6
  - 99.40% at Epoch 7
  - 99.40% at Epoch 8
  - 99.41% at Epoch 9
  - 99.44% at Epoch 10
  - 99.42% at Epoch 11
  - 99.40% at Epoch 12
  - 99.43% at Epoch 13
  - 99.41% at Epoch 14
- Overall testing accuracy is up and we achieved our target during the last 9 epochs. And accuracy overall is high during the later epochs.
- Will try playing with lower parameters

Solution Notebook: `session_07_iteration_10.ipynb`
<br>

<hr>

# Iteration 11

## Target:

- Changed LR scheduler to ReduceLROnPlateau
  - Initial LR = 0.3
  - factor = 0.1
  - patience = 0
  - mode = max
  - metric to be maximized = accuracy, which I'm returning from the train function
- Reduced the number of parameters from 13,808 to 7,750

## Results:

- Parameters = 7,750
- Best training accuracy = 99.14%
- Best testing accuracy = 99.50%

## Analysis:

- Model well within target size, even less than 8,000
- Model is under-fitting as expected.
- We reached our target testing accuracy faster and more consistently
  - 99.44% at Epoch 7
  - 99.44% at Epoch 8
  - 99.44% at Epoch 9
  - 99.50% at Epoch 10
  - 99.46% at Epoch 11
  - 99.49% at Epoch 12
  - 99.46% at Epoch 13
  - 99.48% at Epoch 14
- Overall testing accuracy is up and we achieved our target during the last 8 epochs. And accuracy overall is high during the later epochs.

Solution Notebook: `session_07_iteration_11.ipynb`
<br>

<hr>
