from tensorflow.keras.models import load_model
import numpy as np

model = load_model("modele_parcours.h5")

donnees_entree = [0,0,0,0,0,0]

print("\n~~~~~~~~~Entrer les données pour prediction~~~~~~~~~\n")
donnees_entree[0] = int(input("\tNombre d'heures de dev: "))
donnees_entree[1] = int(input("\tNombre d'heures sur les reseaux: "))
donnees_entree[2] = int(input("\tNote en stat: "))
donnees_entree[3] = int(input("\tInteret pour le design: "))
donnees_entree[4] = int(input("\tNombre de logiciels utiliés: "))
donnees_entree[5] = int(input("\tNombre d'heures de lecture de journal economique: "))

nouvel_etudiant = np.array([donnees_entree])

prediction = model.predict(nouvel_etudiant)
classe_predite = np.argmax(prediction)

parcours = ['DA2I', 'AES', 'RPM']
print("Prédiction :", parcours[classe_predite])