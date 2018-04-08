## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

# Science des Données & Apprentissage Statistique

### [Lire plus...](http://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-lm-Intro-Stat_SD.pdf)

Schématiquement, la **Science des Données** est définie autour d'une *agrégation de compétences* en Informatique (langage comme [R](href="https://cran.r-project.org/) et [Python](https://www.python.org/) , gestion des données, calcul parallèle, *cloud*,...), Statistique (exploration, estimation test, modélisation, prévision) , apprentissage automatique (prévision), Mathématiques (probabilités, optimisation, analyse fonctionnelle, graphes...). 

Son **apprentissage** est acquis par l'intermédiaire de scénarios d'analyse de données réelles, ou *tutoriel*, présentés sous forme de *calepins* ([*jupyter notebooks*](http://jupyter.org/)) en [R](href="https://cran.r-project.org/) ou [Python](https://www.python.org/). Voir à ce sujet le [livre de référence](https://www.inferentialthinking.com/) du cours [*Fondations of Data Science*](http://data8.org/) de l'UC Berkley.

Cette **pratique** est **indispensable** mais masque les *aspects théoriques* (mathématiques, statistique, probabilités, optmisation): une *formule* est remplacée par un commande ou fonction en Python ou R, une *démonstration* par l'exécution d'exemples dans un calepin.

Pour offrir de la *profondeur*, plus de compréhension, à cette (auto)-formation, les calepins renvoient (liens hypertextes) systématiquement à des **vignettes "théoriques"**  du site [wikistat.fr](http://wikistat.fr/) exposant en détail (cours) les méthodes et algorithmes concernés.

Il ne s'agit pas simplement de pouvoir exécuter une méthode, un algorithme, il est important d'en **comprendre les propriétés**, conditions d'utilisation et limites.

# Saison 4 [*Technologies des Grosses Data*](https://github.com/wikistat/Ateliers-Big-Data) 
#<a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/390px-Python_logo_and_wordmark.svg.png" style="max-width: 120px; display: inline" alt="Python"/></a> <a href="http://spark.apache.org/"><img src="http://spark.apache.org/images/spark-logo-trademark.png" style="max-width: 80px; display: inline" alt="Spark"/> </a> <a href="https://www.tensorflow.org/"><img src="https://avatars0.githubusercontent.com/u/15658638?s=200&v=4" style="max-width: 40px; display: inline" alt="TensorFlow"/></a>  <a href="https://keras.io/"><img src="https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png" style="max-width: 100px; display: inline" alt="Keras"/></a>

### [Introduction plus détaillée...](http://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-Intro-AtelierGD.pdf)

Si les données ne sont pas *grosses* à devoir être *distribuées*, un usage classique de Python voire R s'avère plus efficace pour une *phase d'apprentissage*. En revanche, la phase de préparation de données massives (*data munging*), en flux ou pas, gagne beaucoup à être opérée dans un environnement distribué (*Hadoop*) utilisant *Spark*, notamment *via* l'API `PySpark` (cf. [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099)). 

Dans tous les cas, l'apprentissage de modèles complexes (apprentissage épais ou *deep learning* avec `keras- tensorFlow`, `XGBoost`...) nécessite des moyens de calcul massivement parallèles (*e.g. GPU, cloud*). D'où l'importance pour le statisticien / *data scientist* de se former aux technologies afférentes.

## Objectifs

Cette saison est consacrée au **passage à l'échelle** pour l'analyse de *grosses* *data*, donc distribuées sur plusieurs machines (*Hadoop*) ou de données en *grande dimension* (signaux, images). L'objectif est de comparer les performances des principaux environnements ou architectures plus ou moins bien adaptées à l'analyse de données massives en fonction du but visé: préparation des donnés, exploration, apprentissage, prévision.

## Prérequis
Avoir acquis les compétences des saisons précédentes ou y revenir:

- [Initiation à R](https://github.com/wikistat/Intro-R)
- [Initiation à Python](https://github.com/wikistat/Intro-Python)
- Formation aux [outils Statistiques de base](https://github.com/wikistat/StatElem)
- [Exploration Statistique pour la Science des Données](https://github.com/wikistat/Exploration). Cete saison intègre les algorithmes d'apprentissage non-supervisé (*clustering*).
- [Apprentissage Automatique / Statistique](https://github.com/wikistat/Apprentissage)

## Déroulement de l'UF: Technologies pour l'analyse de données massives
L'apprentissag est divisée en 5 ateliers ciblant des technologies et/ou des types de données particuliers.

- **Épisode 1** *Spark*: [Complément](https://github.com/wikistat/Intro-Python/blob/master/Cal4-PythonProg.ipynb) de programmation en Python puis [Initiation](https://github.com/wikistat/Intro-PySpark) à Spark *via* l'API `PySpark`.
- **Épisode 2** *MLlib*: pratique de la librairie MLlib de Spark pour l'apprentissage sur données distribuées. Comparaison des performances en R, Python et MlLib sur des cas d'usage
   - [`MNIST`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/MNIST) reconnaissance de caractères manuscrits avec R, `Scikit-learn` et MLlib. Laisser la version avec `keras` pour l'épisode 4.
   - [`MovieLens`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/MovieLens) Système de recommandation de films avec R (`softImpute`) et MLlib (*non negative matrix factorisation*).
- **Épisode 3** [`Cdiscount`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/Cdiscount) catégorisation de produits (*text mining*): Scikit-learn vs. MLlib.
- **Épisode 4** *Reconnaissance d'images*; apprentissage épais avec `TensorFlow` utilisé *via* `Keras`. Deux cas d'usage: [`MNIST`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/MNIST) et [cats *vs.* dogs](https://github.com/wikistat/Ateliers-Big-Data/tree/master/CatsVSDogs) avec transfert de l'apprentissage du  réseau `inception` de Google. 
- **Épisode 5 en chantier** [*Cloud computing*]() accéder à AWS pour passer à l'échelle; installer un container, transférer un code de calcul, l'exécuter.

Une synthèse des résultats obtenus dnas les 3 cas d'usage (MNIST, MovieLens, Cdiscount) est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

Chaque année, l'**évaluation** des compétences acquises est basée sur la participation (résultats et soutenance orale) des étudiants à un [défi grosses data](https://defibigdata2019.insa-toulouse.fr/) dont l'objet est la construction d'une meilleure prévision par apprentissage sur un jeu complexe de données.



