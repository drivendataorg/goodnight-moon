# Solution -  Speech text comparison as speech classification

Username: dylanliu

## Summary

This code base is the inference code base plus training code, and includes 3 parts: data preprocessing, training, and inference.

# Setup

1. Install Python 3.10.14 and Pytroch 2.4.0+CUDA if you don't have them on your machine. Note that the training environment and the inference environment are different, especially the python version.   
How to install Python: https://docs.python.org/3/using/unix.html#on-linux  
How to install Pytroch: https://pytorch.org/get-started/previous-versions/  

2. Install the required Python packages:
`pip install -r requirements.txt`

# Run data preprocessing
First, move the training dataset to the "data/" folder, that is, the path of "train_metadata.csv" in the dataset should be "data/train_metadata.csv". 

Run data preprocessing by the command line:  
`python preprocess.py`  

After preprocessing, a folder "tts_data" can be found under the code folder.  

# Run training
Run training by the command line:   
`python train.py`  

After training, the trained models can be found under the code folder.   

You can see that the training code for each model is independent, and train.py just uses them uniformly. You can select the models to be trained in train.py, and split the training process for parallel training to make full use of the GPU, because these models are not large and will not take up too much memory. You can also train a fold of a model separately. For example, train the fold 0 of "model3mapd":  
`python train_model3mapd.py --fold 0 --n_folds 6`  
The fold parameter in this command is the fold number (0 to 5, a total of 6 folds). n_folds is the total number of folds, which must be 6, otherwise there will be an error later.  

After training a fold of a model, you will see a model file consisting of the model name and fold in the current directory, such as "model3mapd_0.pth" where "model3mapd" is the model name and 0 is the model fold.  
After training 10*6 models, move all model files to the "assets/" folder, and then run:  
`python check_models.py`  
to check the cv scores (log_loss) of all models.

As I said in the model documentation, gradient vanishing occurs with a very small probability during the competition. Although the models I chose did not experience gradient vanishing, when you check the cv scores of the model, if the loss details of a fold are higher than the loss of other folds, such as the loss of a fold is greater than 0.28, and the loss of other folds is 0.24-0.22, you need to retrain this fold using the above method. Hopefully this will not happen.  

# Run inference
Before inference, run:   
`python get_config.py`  
to get the weights of the model ensemble and generate the configuration file "model_configs.pkl".

Then move the current code base to the competition environment, and then run
`python infer.py`
for inference. It's running logic is consistent with "main.py" during the competition, and finally generates "submission.csv".