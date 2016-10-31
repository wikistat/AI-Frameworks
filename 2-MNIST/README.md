###[**INSA de Toulouse**](http://www.insa-toulouse.fr/fr/index.html), [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

## [Ateliers: Technologies des Données Massives](https://github.com/wikistat/Ateliers-Big-Data) avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume** des méthodes d'apprentissage; **processus qui transforme un statisticien en *Data Scientist*.** 


# Reconnaissance de caractères  ([MNIST](http://yann.lecun.com/exdb/mnist/)) 


**Résumé** Présentation du problème de reconnaissance de caractères manuscrits ([MNIST DataBase](http://yann.lecun.com/exdb/mnist/)) à partir d’images numérisées. L’objectif
n’est pas la recherche de la meilleure prévision mais une comparaison des performances de différentes technologies ou librairies (R, Scikit-learn, Spark MLlib)

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

## Introduction
L'objectif général est la construction d'un meilleur modèle de reconnaissance de chiffres manuscrits. Ce problème est ancien (zipcodes) et sert souvent de base pour la comparaison de méthodes et d'algorithmes d'apprentissage. Le site de Yann Le Cun: [MNIST](http://yann.lecun.com/exdb/mnist/) DataBase, est à la source des données étudiées, il décrit précisément le problème et les modes d'acquisition. Il tient à jour la liste des publications proposant des solutions avec la qualité de prévision obtenue. Ce problème a également été proposé comme sujet d'un concours [Kaggle](https://www.kaggle.com/competitions) mais sur un sous-ensemble des données. 

De façon très schématique, plusieurs stratégies sont développées dans une vaste littérature sur ces données.  

- Utiliser une méthode classique (k-nn, random forest...) sans trop raffiner mais avec des temps d'apprentissage rapide conduit à un taux d'erreur autour de 3\%.
* Ajouter  ou intégrer un pré-traitement des données permettant de recaler les images par des distorsions plus ou moins complexes.
* Construire une mesure de distance adaptée au problème, par exemple invariante par rotation, translation, puis l'intégrer dans une technique d'apprentissage classique comme les $k$ plus proches voisins.
* Utiliser une méthode plus flexibles (réseau de neurones "profond") avec une optimisation fine des paramètres.

L'objectif de cet atelier n'est pas de concurrencer les meilleurs prévisions mais de comparer sur des données relativement volumineuses les performances de différents environnements technologiques et librairies.  Une dernière question est abordée, elle concerne l'influence de la taille de l'échantillon d'apprentissage sur le temps d'exécution ainsi que sur la qualité des prévisions.

## Tutoriels

Trois environnements sont testés, chacun dans un calepin.

- [`Atelier-MNIST-R`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/2-MNIST/Atelier-MNIST-R.ipynb) en [R](https://cran.r-project.org/).
- [`Atelier-MNIST-python`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/2-MNIST/Atelier-MNIST-python.ipynb) en Python avec la librairie [scikit-learn](http://scikit-learn.org/stable/)
- [`Atelier-MNIST-python`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/2-MNIST/Atelier-MNIST-pyspark.ipynb) utilise la librairie [MLlib](http://spark.apache.org/mllib/) de [Spark](http://spark.apache.org) à l'aide de l'API `pyspark` et donc finalement en langage Python.

**Remarque**, les meilleures résultats actuellement obtenus font appel à des modèles invariants par transformation: *scattering* ou apprentissage profond avec réseaux de neurones  *convolutionnels* dont de nombreuses librairies proposent des implémentations : keras, theano, lasagne, tensorflow, torch... mais nécessitant des moyens techniques (cluster, cartes GPU) conséquents.


