import os
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


def load_images_from_folder(folder, target_shape=None):
    images = []
    labels = []
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith('.jpeg'):
                    img = cv2.imread(os.path.join(subfolder_path, filename))
                    if img is not None:
                        if target_shape is not None:
                            img = cv2.resize(img, target_shape)
                        images.append(img)
                        labels.append(subfolder)
                        print('Labels \n', labels)
                    else:
                        print(f"Warning: Unable to load {filename}")
    return images, labels



data_folder = './dataset'
test_folder = './test'

images, labels = load_images_from_folder(data_folder, target_shape=(200, 200))

#Se convertesc pozele la scala de gri
image_data = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).flatten() for image in images]
image_data = np.array(image_data)

#Standardizarea datelor pentru a asigura o distribuție normală.
scaler = StandardScaler()
scaled_data = scaler.fit_transform(image_data)




X_train, X_test, y_train, y_test = train_test_split(scaled_data, labels, test_size=0.2, random_state=42)


knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)


y_pred_knn = knn_classifier.predict(X_test)




nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)


y_pred_nb = nb_classifier.predict(X_test)



test_images, test_labels = load_images_from_folder(test_folder, target_shape=(200, 200))

for i, test_image in enumerate(test_images):
    test_image_data = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY).flatten()
    scaled_test_data = scaler.transform([test_image_data])
    

    
    test_prediction_knn = knn_classifier.predict(scaled_test_data)
    print(f"Test Image {i + 1} - KNN Predicted Class: {test_prediction_knn[0]}")

   
    test_prediction_nb = nb_classifier.predict(scaled_test_data)
    print(f"Test Image {i + 1} - Naive Bayes Predicted Class: {test_prediction_nb[0]}")

   
    plt.figure()
    plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
    plt.title(f"Test Image {i + 1} - KNN Predicted Class: {test_prediction_knn[0]}, Naive Bayes Predicted Class: {test_prediction_nb[0]}")
    plt.axis('off')
    plt.show()

    models = ['GaussianNB', 'K-Nearest Neighbors']

accuracy_scores_test = [
    accuracy_score(y_test, y_pred_nb),
    accuracy_score(y_test, y_pred_knn),
]


plt.bar(models, accuracy_scores_test, color=['blue','red'])
plt.xlabel('Algoritmi')
plt.ylabel('Acuratețe')
plt.title('Compararea Acurateții între Naive Bayes și K-Nearest Neighbors (Testare)')
plt.ylim(0, 1) 
plt.show()

print('Confusion matrix NB:')
print(confusion_matrix(y_test, y_pred_nb))
print('Confusion matrix KNN:')
print(confusion_matrix(y_test, y_pred_knn))
