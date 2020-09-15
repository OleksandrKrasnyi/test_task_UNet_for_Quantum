Realization of some semantic segmentation model with UNet architecture using Keras TensorFlow.

Deployed exploratory is fully represented in unet_model.ipynb
Consists of next parts:
	0. Required Imports
	1. Data Preprocessing
	2. Hyperparameters
	3. Data Preview
	4. Different Convolutionla Blocks
	5. UNet Model
	6. Training the Model
	7. Testing the Model
	
How to run it:
	1. Get dataset from link in dataset.txt
	2. Download dataset.rar and extract dataset folder.
	3. Open unet_model.ipynb
	4. Run cells in unet_model.ipynb consistently.
	
	Functionality is also represented in 3 python files:
		- unet_model_core.py - includes everything required to activate the model;
		- Train.py - implements training (may take some time) of the model;
		- Predict_mask.py - tests the model using weight got during training.

About the Model:
	The main part of model is convolutional blocks which were supported by pooling layers (4. Different Convolutional Blocks). They are used in 5. UNet Model.
	The blocks are mixed in a way which corresponds to UNet architecture.
	Model training may take some time.
	It is made of simplest tensorflow.keras constructions and doesn't rely on keras-segmentation.
	
	Model Architecture:
		Downsampling Path: 
			It consists of two 3x3 convolutions (unpadded convolutions),
			each followed by a rectified linear unit (ReLU) and a 2x2 max pooling operation with stride 2 for downsampling.
			At each downsampling step the number of feature channels doubles.
		Upsampling Path: 
			Every step in the expansive path consists of an upsampling of the feature map followed by a 2x2 convolution (“up-convolution”),
			a concatenation with the correspondingly feature map from the downsampling path, and two 3x3 convolutions, each followed by a ReLU.
		Skip Connection: 
			The skip connection from the downsampling path are concatenated with feature map during upsampling path.
			These skip connection provide local information to global information while upsampling.
		Final Layer: 
			At the final layer a 1x1 convolution is used to map each feature vector to the desired number of classes.