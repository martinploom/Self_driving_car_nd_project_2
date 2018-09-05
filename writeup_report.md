# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report_materials/classes_histogram.png "Histogram of classes"
[image2]: ./report_materials/initial_image.png "Initial image"
[image3]: ./report_materials/normalized_image.png "Normalized image"
[image4]: ./report_materials/gray_image.png "Grayscale image"
[image5]: ./report_materials/new_images.png "Traffic signes downloaded from the internet"
[image6]: ./report_materials/placeholder.png "Traffic Sign 3"
[image7]: ./report_materials/placeholder.png "Traffic Sign 4"
[image8]: ./report_materials/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. A Writeup / README that includes all the rubric points and how I addressed each one. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/martinploom/Self_driving_car_nd_project_2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. 

I used the python builtin library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34 799.
* The size of the validation set is 4410
* The size of test set is 12 630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a histogram showing how many examples there are in the training set for each unique class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize and mean center the image data. To do so I subtracted 128 from the RGB values and then divided the result by 128 to get values in the range of -1 to 1.

Here is an example of a traffic sign image before 

![alt text][image2] 

and after normalizing ...

![alt text][image3] 

After normalizing to improve the results even futher I decided to convert the image to grayscale.

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         			|     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         			| 32x32x3 RGB image   							| 
| Convolution 5x5     		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU						|												|
| Max pooling	      		| 2x2 stride, outputs 14x14x6	 				|
| Convolution 5x5	    	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU						| 												|
| Max pooling				| 2x2 stride, outputs 5x5x16					|
| Flatten					| Output 400									|
| Fully connected			| Output 120									|
| RELU						|												|
| Dropout					| Keep probability 0.5							|
| Fully connected			| Output 84										|
| RELU						|												|
| Dropout					| Keep probability 0.5							|
| Fully connected / output	| Output 43										|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a LeNet architecture and modified it with couple of dropout layers to stop the model from underfitting and get better results. For optimizer an Adam optimizer was used. 
 Batch size was 64 as from playing around with it to smaller or bigger side while keeping other parameters the same the results got worse. 
Number of epochs was selected regarding the train and validation set scores. Both of them had an increasing trend up to 15 epochs and the results were promising so I stopped there. 
Learning rate hypterparameter was started out from 0.001 and I modified it to both sides but once again the results didn't improve so I stuck with this learning rate.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.1%
* validation set accuracy of 96.1% 
* test set accuracy of 93.0%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

First was LeNet. The accuracy on the validation set was ~88%.
* What were some problems with the initial architecture?

The initial architecture was underfitting the data.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

First adjustment was to normalize and mean center the data which boosted the accuracy to ~90% level.

After that I converted the images to grayscale, which gave ~90.5% accuracy for the validation set.

Then I tried flipping images, but it didn't boost the results so I decided to remove it from the architecture.

Next, I tried dropout layer before last layer, result ~93.5%. As the initial goal is achieved, but things got interesting I tried even further.

Then I increased epochs number to 15 and added monitoring for training accuracy and saw that training accuracy kept rising, but validation accuracy started fluctuating so the model was overfitting. So I added another dropout and got better results in the form of 95.2%

Next up I changed batch size to 64.  The result was 95.8% and the validation and training accuracy kept rising in general with minor fluctuations in the middle.

After that changed learning rate to 0.0009 and the results weren't so good - 94.6%.

Tried changing learning rate to 0.0011 and the result was 95.8 with fluctuating in the middle which indicates a bit for overfitting.

Went back to learning rate 0.001 and achieved 96.1% accuracy and the training and validation set kept increasing simultanously.

* Which parameters were tuned? How were they adjusted and why?

I tuned epochs, batch size and learning rate. I changed one at a time and monitored the training and validation set accuracies to see if things got better or worse.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

LeNet architecture was selected as it has proven to work in a similar task. Convolutiona layer might be a good choice as it selects some features of the image that it 
detects and as the depth of layers increases, each layers focuses on some feature and as more and more convolution layers are connected more complex features are 
filtered out.

If a well known architecture was chosen:
* What architecture was chosen?

LeNet architecture was chosen.

* Why did you believe it would be relevant to the traffic sign application?

The LeNet was the first successful convolutional network regarding image classification and it is a good simple platform to build on. Other 
wellknown networks have been built based on LeNet as well, such as AlexNet.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

All in all I'd say that the model is working pretty well. The training and validation set accuracies are good and throughout 
the epochs they are both rising which indicates that the model is training well and not under or over fitting the dataset. The test set accuracy 
is a bit lower, but not by much.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image5] 


The first image might be difficult to classify because it has so few images in the dataset and network might be biased not to detect it.
 On the other hand the 4th image (30 kmh speed limit) sign is well reprsented in the dataset. The rest of the images are somewhat in 
 the middle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        					|     Prediction	        					| 
|:-----------------------------------------:|:---------------------------------------------:| 
| Speed limit (20 kmh)      				| Speed limit (30 kmh)   						| 
| Right-of-way at the next intersection  	| Right-of-way at the next intersection 		|
| No entry									| No entry										|
| Speed limit (30 kmh)	      				| Speed limit (30 kmh)					 		|
| Road work									| Beware of ice/snow      						|
| Ahead only								| Ahead only									|
| Speed limit (100 kmh)						| Speed limit (100 kmh)							|
| No passing								| No passing									|


The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. This doesn't compares favorably 
to the accuracy on the test set, but as there were only 8 images a results are easy to not match up as the test set was ~12 000 
images large and the result can stabilize and couple of pictures can't affect the result so much as they can if there are only 
8 images in total.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located under the "Output Top 5 Softmax Probabilities For Each Image Found on the Web"
section of the Ipython notebook.

For the first image, the model is relatively sure that this is a 30 kmh speed limit (probability of 0.997), but the image does contains a 
20 kmh speed limit sign. This images was selected on purpose as from the training set it can be seen that the 20 kmh sign has very few examples 
and I didn't augment data to even out these to see how the model behaves like this. 

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .997         			| Speed limit (30 kmh)   						| 
| .003     				| Speed limit (20 kmh) 							|
| .000					| End of speed limit (80km/h)					|
| .000	      			| General caution				 				|
| .000				    | Right-of-way at the next intersection 		|


For the second image, the model is sure that this is a Right-of-way at the next intersection (probability of 1.0), and the image does contain a 
Right-of-way at the next intersection sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| Right-of-way at the next intersection  		| 
| .000     				| Beware of ice/snow 							|
| .000					| Pedestrians									|
| .000	      			| Double curve				 					|
| .000				    | Roundabout mandatory					 		|

For the third image, the model is relatively sure that this is a No entry (probability of 0.804), and the image does contain 
that sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .804         			| No entry								  		| 
| .164     				| Priority road 								|
| .032					| No passing									|
| .000	      			| End of no passing				 				|
| .000				    | Ahead only							 		|

For the fourth image, the model is sure that this is a 30 kmh speed limit (probability of 1.0), and the image does contain 
that sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| Speed limit (30km/h)  						| 
| .000     				| Speed limit (20km/h)							|
| .000					| End of speed limit (80km/h)					|
| .000	      			| Speed limit (70km/h)				 			|
| .000				    | Speed limit (80km/h)					 		|

For the fifth image, the model is quite sure that this is a Beware of ice/snow (probability of .973), but the image contains 
Road work sign. It is a bit weird as the Road work sign is pretty well represented in the dataset and on the other hand the 
Beware of the ice/snow is not so well represented so it might be the fact that they both have triangular shape and some object
in the middle of it that confused the model.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .973        			| Beware of ice/snow  							| 
| .026     				| Right-of-way at the next intersection			|
| .001					| End of no passing								|
| .000	      			| Slippery road				 					|
| .000				    | Dangerous curve to the right					|

For the sixth image, the model is sure that this is a Ahead only (probability of 1.0), and the image does contain 
that sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| Ahead only  									| 
| .000     				| Go straight or right							|
| .000					| Yield											|
| .000	      			| No passing						 			|
| .000				    | Speed limit (60km/h)					 		|

For the seventh image, the model is sure that this is a 100 kmh speed limit (probability of .998), and the image does contain 
that sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .998        			| Speed limit (100km/h)  						| 
| .002     				| No passing for vehicles over 3.5 metric tons	|
| .000					| Speed limit (80km/h)							|
| .000	      			| Speed limit (120km/h)				 			|
| .000				    | Vehicles over 3.5 metric tons prohibited 		|

For the eighth image, the model is sure that this is a No passing (probability of 1.0), and the image does contain 
that sign.

The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| No passing  									| 
| .000     				| Vehicles over 3.5 metric tons prohibited		|
| .000					| No passing for vehicles over 3.5 metric tons	|
| .000	      			| Ahead only				 					|
| .000				    | End of no passing					 			|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


