## Traffic Sign Classification
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, we use convolutional neural networks to classify traffic signs. We train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

[//]: # (Image References)

[image0]: ./dataset_histogram.png "Dataset Histogram"
[image1]: ./images_from_web/dangerous_curve_right.png "Dangerous Curve to the Right"
[image2]: ./images_from_web/keep_right.png "Keep Right"
[image3]: ./images_from_web/no_passing.png "No Passing"
[image4]: ./images_from_web/priority_road.png "Priority Road"
[image5]: ./images_from_web/roundabout.png "Roundabout"

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset

Here is an exploratory visualization of the data set. It is a histogram showing how the data is distributed between 43 labels

![Dataset Histogram][image0]

### Design and Test a Model Architecture

#### 1. Preprocessing

As part of the preprocessing step, I combined all the dataset and reshuffled to generate the training, validation and testing datasets using the `train_test_split` function. Follwed by normalizing the data so the data has close to zero mean and equal variance. In future, I would like to experiment with data augmentation to generate additional data. As seen in the above histogram, the distribution of data between various labels is uneven. We could use data augmentation to balance it out and that would help produce even better results.

#### 2. Learning Model

I started with the default _LeNet-5_ architecture and played around with adding Dropout layers which didn't seem to help much with the accuracy. My final model consisted of the following layers:

| Layer         		|     Description	        					          | 
|:-----------------:|:-------------------------------------------:| 
| Input         		| 32x32x3 RGB image   						            | 
| Convolution 5x5  	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					    |												                      |
| Max pooling	      | 2x2 stride, valid padding, outputs 14x14x6  |
| Convolution 5x5   | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU					    |												                      |
| Max pooling	      | 2x2 stride, valid padding, outputs 5x5x16		|
| Flatten		        | outputs 400        			        						|
| Fully Connected		| outputs 120              									  |
| RELU					    |						        						              |
| Fully Connected		| outputs 84        									        |
| RELU					    |												                      |
| Fully Connected		| outputs 43        									        |
| Softmax				    |       									                    |

#### 3. Trainining the model

To train the model, I used an _adam_ optimizer, with a batch size of _256_, learning rate of _0.002_, running it for _10 epochs_.

#### 4. Approach Taken

I followed an iterative approach converging to an accuracy > 0.93. Since I already had an implementation for the LeNet-5 architecture, I just went ahead with that in the beginning by just modifying the image channel depth to _3_. Later, more fully connected layers were added to the bottom and a dropout layer with a 0.9 dropout was also included in between these fully connected layers. But this did not help increase the accuracy, so I reverted back to the original architecture. What helped was shuffling and normalizing of data in the pre-processing stages. The learning rate hyperparameter was increased from _0.001_ to _0.002_ for faster convergence since I observed that with the default learning rate, it required more number of epochs to reach the optimum accuracy.

My final model results were:
* validation set accuracy of 97.4%
* test set accuracy of 97.5%

Since, the validation set accuracy is very close to the test set accuracy, the model is working well. In future, data augmentation coupled with a droupout layer should help increase the accuracy.

### Testing the Model on New Images

Below are five German traffic signs that I found on the web:

![Dangerous Curve to the Right][image1] ![Keep Right][image2] ![No Passing][image3] 
![Priority Road][image4] ![Roundabout][image5]

The last image might be difficult to classify because of its low contrast ratio.

Here are the results of the prediction:

| Image			                    |     Prediction	        					| 
|:-----------------------------:|:---------------------------------:| 
| Dangerous curve to the right  | Dangerous curve to the right      |
| Keep right     			          | Keep right 								      	|
| No passing					          | No passing											  |
| Priority road	      		      | Priority road					 				    |
| Roundabout mandatory			    | Roundabout mandatory      				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 97.2%

#### 3. Prediction Probabilities

The code for making predictions on my final model is located in the 9th cell of the Ipython notebook. For the first image, the model is sure that this is a _Dangerous curve to the right_ sign (probability of 1.0), and the image does contain a stop sign. The last image was relatively difficult to classify due to the low contrast. But the model could predict it with a probability of _0.94_. The top five soft max probabilities were:

| Probability  |     Prediction	        					| 
|:------------:|:--------------------------------:| 
| 1.00         | Dangerous curve to the right 		| 
| 1.00    		 | Keep right									      |
| 1.00				 | No passing									      |
| 1.00      	 | Priority road			 				      |
| 0.95				 | Roundabout mandatory							|
