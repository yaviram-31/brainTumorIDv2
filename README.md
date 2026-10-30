brainTumorIDv2

A new iteration of brainTumorID, now utilizing ML for multi-class classification

What this model improves on from the previous one:

    1. Full deep-learning architecture using PyTorch

    2. Multi-class classification that is more accurate than the binary classification method used in brainTumorID

    3. Utilizing the full strength of Python, allowing for a wider range of applications, modules, and a general increase in versatility from MATLAB

What this program does:

    1. Takes a .jpg file as input

    2. Feeds it into the pre-trained CNN (for reference, training took place in "train.ipynb") and makes a prediction
    
    3. Using Matplotlib, the picture is displayed with the prediction as the title, as well as the heat map from the second convolutional layer (pytorch-gradcam)

MRI scan dataset was found at: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Disclaimer: This is for education and research only. It is not a medical diagnostic tool.