## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

## [Ateliers: Technologies des Données Massives](https://github.com/wikistat/Ateliers-Big-Data) 


<a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/390px-Python_logo_and_wordmark.svg.png" style="max-width: 120px; display: inline" alt="Python"/></a> <a href="http://spark.apache.org/"><img src="http://spark.apache.org/images/spark-logo-trademark.png" style="max-width: 80px; display: inline" alt="Spark"/> </a> <a href="https://www.tensorflow.org/"><img src="https://avatars0.githubusercontent.com/u/15658638?s=200&v=4" style="max-width: 40px; display: inline" alt="TensorFlow"/></a>  <a href="https://keras.io/"><img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" style="max-width: 100px; display: inline" alt="Keras"/></a>

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume** des méthodes d'apprentissage; **processus qui transforme un statisticien en *Data Scientist*.** 


# Reconnaissance de caractères  ([MNIST](http://yann.lecun.com/exdb/mnist/)) 


**Résumé** Présentation du problème de reconnaissance de 
caractères manuscrits ([MNIST 
DataBase](http://yann.lecun.com/exdb/mnist/) à partir d’images 
numérisées. L’objectif, dans un premier temps, n’est pas la 
recherche de la meilleure prévision mais une comparaison des 
performances de différentes technologies ou librairies (R, 
Scikit-learn, Spark MLlib). De "meilleures" classifications sont 
obtenues par apprentissage épais dans l'épisode 4.

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

## Introduction
L'objectif général est la construction d'un meilleur modèle de reconnaissance de chiffres manuscrits. Ce problème est ancien (zipcodes) et sert souvent de base pour la comparaison de méthodes et d'algorithmes d'apprentissage. Le site de Yann Le Cun: [MNIST](http://yann.lecun.com/exdb/mnist/) DataBase, est à la source des données étudiées, il décrit précisément le problème et les modes d'acquisition. Il tient à jour la liste des publications proposant des solutions avec la qualité de prévision obtenue. Ce problème a également été proposé comme sujet d'un concours [Kaggle](https://www.kaggle.com/competitions) mais sur un sous-ensemble des données. 

De façon très schématique, plusieurs stratégies sont développées dans une vaste littérature sur ces données.  

- Utiliser une méthode classique (k-nn, random forest...) sans trop raffiner mais avec des temps d'apprentissage rapide conduit à un taux d'erreur autour de 3\%.
* Ajouter  ou intégrer un pré-traitement des données permettant de recaler les images par des distorsions plus ou moins complexes.
* Construire une mesure de distance adaptée au problème, par exemple invariante par rotation, translation, puis l'intégrer dans une technique d'apprentissage classique comme les $k$ plus proches voisins.
* Utiliser une méthode plus flexibles (réseau de neurones épais) avec une optimisation fine des paramètres.

L'objectif de cet atelier est de comparer sur des données relativement volumineuses les performances de différents environnements technologiques et librairies.  Une dernière question est abordée, elle concerne l'influence de la taille de l'échantillon d'apprentissage sur le temps d'exécution ainsi que sur la qualité des prévisions.

## Tutoriels

Quatre environnements sont testés, comparés, chacun dans un calepin.

- [`Atelier-R-MNIST`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/MNIST/Atelier-R-MNIST.ipynb) en [R](https://cran.r-project.org/).
- [`Atelier-scikit_learn-MNIST`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/MNIST/Atelier-scikit_learn-MNIST.ipynb) en Python avec la librairie [scikit-learn](http://scikit-learn.org/stable/).
- [`Atelier-pyspark-MNIST`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/MNIST/Atelier-pyspark-MNIST.ipynb) utilise la librairie [MLlib](http://spark.apache.org/mllib/) de [Spark](http://spark.apache.org) à l'aide de l'API `pyspark` et donc finalement en langage Python.
- [`Atelier-keras-MNIST`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/MNIST/Atelier-keras-MNIST.ipynb) déploie différentes couches de réseaux, dont celles convolutionnelles, *via* [`Keras`](https://keras.io/) au-dessus de [`TensorFlow`](https://github.com/tensorflow/tensorflow).

Exécuter les trois premiers tutoriels lors de l'épisode 2, le 4ème 
lors de l'épisode 3.




