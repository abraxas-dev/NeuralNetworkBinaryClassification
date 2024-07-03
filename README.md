# Neural Network for Binary Classification
This repository contains a neural network project for binary classification implemented from scratch using Python as an exercise for the Deep Learning Specialisation course. The primary purpose of this project was to deepen understanding of neural networks and their application in binary classification. The program includes an interactive window in the terminal that prompts the user for various settings and information required for program execution.

**P.S. This project is not finished yet. Some features and changes are still to come.**

## Project Overview 

### Note
- I achieved 100% accuracy on the training dataset and 82% on the test dataset using a learning rate of **0.05** and **500** iterations. Additionally, the neural network correctly classified each provided picture.
- Use the provided cat0 dataset and images for experimentation. Please note that this dataset is very small, which may lead to overfitting.
- To modify the number of layers/neurons, adjust dims section in the code. Ensure that the number of neurons in the first layer matches the input dimensions of your dataset images.
- To modify the dataset, ensure your dataset is saved in the /datasets/ directory with the .h5 extension.
- Store your images in the /images/ directory.
- Note that there is no error handling implemented; incorrect input may lead to unexpected behavior.
- Tested on Unix Systems.

### Usage
1. Clone the repository
```
git clone https://github.com/abraxas-dev/NeuralNetworkBinaryClassification.git
```
2. Navigate to the project directory
```
cd ./NeuralNetworkBinaryClassification/
```
3. Create and activate a virtual environment:
```
python3 -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
4. Install the dependencies
```
pip3 install -r requirements.txt
```
5. Run the application
```
python3 model.py
```

### Example output with the provided dataset and images

<p align="center">
  <img src="example%20output/nn-image1.png" width="600" />
  <img src="example%20output/nn-image2.png" width="600" />
</p>
