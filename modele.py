import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Chaque ligne du tableau représente un étudiant avec 6 caractéristiques, 
# qui va permettre de detterminer si l'etudiant est en DA2I, RPM ou AES
# Les caracteristique representé par ces données sont: 
#       nb d'heures de dev, nb d'heures sur les reseaux, note en stat, 
#       interet pour le design, nb de logiciels utilisés, lecture de journal economique


X = np.array([
    [15, 5, 10, 2, 3, 0],  # DA2I
    [12, 3, 12, 1, 2, 1],  # DA2I
    [18, 2, 11, 3, 4, 0],  # DA2I
    [2, 15, 16, 7, 2, 1],  # AES
    [3, 18, 18, 6, 1, 1],  # AES
    [4, 12, 17, 8, 2, 1],  # AES
    [5, 20, 9, 9, 5, 0],   # RPM
    [6, 22, 8, 10, 6, 0],  # RPM
    [7, 25, 10, 10, 7, 0], # RPM
    [14, 4, 12, 2, 3, 0],  # DA2I
    [3, 16, 19, 7, 1, 1],  # AES
    [5, 21, 9, 9, 6, 0],   # RPM
    [13, 3, 11, 1, 4, 0],  # DA2I
    [2, 14, 17, 7, 2, 1],  # AES
    [6, 23, 8, 10, 6, 0],  # RPM
])

#Sorties attendues : 0 = DA2I, 1 = AES, 2 = RPM
y = np.array([0,0,0, 1,1,1, 2,2,2, 0,1,2, 0,1,2])

Y = to_categorical(y, num_classes=3)

model = Sequential()
model.add(Dense(10, input_shape=(6,), activation='relu'))
model.add(Dense(8, activation='relu'))                   
model.add(Dense(3, activation='softmax'))                  

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Entraînement

model.fit(X, Y, epochs=150, verbose=1)
model.save("modele_parcours.h5")
