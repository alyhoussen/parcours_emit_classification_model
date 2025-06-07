# Major Classifier Model for EMIT Students

## Overview

This project is a machine learning model designed to classify EMIT students into their respective majors: **DA2I** (Computer Science), **AES** (Management), or **RPM** (Multimedia). The classification is based on a set of student-specific features, such as the number of hours spent on development, network activities, statistics grades, interest in design, number of software used, and economic journal reading habits.

## How It Works

- The model is a neural network built with TensorFlow/Keras.
- It takes 6 input features for each student:
  1. Number of hours of development
  2. Number of hours spent on networks
  3. Grade in statistics
  4. Interest in design
  5. Number of software used
  6. Number of hours reading economic journals
- The model outputs a prediction for the most likely student major: **DA2I**, **AES**, or **RPM**.

## Usage Example

To use the trained model for classification, you can run the provided `tester_modele.py` script. This script loads the model and prompts for input values:

```bash
python tester_modele.py
```

You will be prompted to enter six values:
1. Number of hours of development
2. Number of hours spent on networks
3. Grade in statistics
4. Interest in design
5. Number of software used
6. Number of hours reading economic journals

Example interaction:
```
~~~~~~~~~Entrer les données pour prediction~~~~~~~~~

	Nombre d'heures de dev: 15
	Nombre d'heures sur les reseaux: 5
	Note en stat: 10
	Interet pour le design: 2
	Nombre de logiciels utiliés: 3
	Nombre d'heures de lecture de journal economique: 0
Prédiction : DA2I
```

The model will output the predicted major for the student.

## Model Details

- The training data consists of examples for each major, with each example being a vector of six values.
- The neural network is composed of two hidden layers (with 10 and 8 units respectively) and a softmax output layer for three classes.
- The model is trained using categorical cross-entropy loss and the Adam optimizer.