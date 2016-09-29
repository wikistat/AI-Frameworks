# Ateliers: Technologies des Données Massives avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume**; processus qui transforme un statisticien en *Data Scientist*. 


## Catégorisation des Produits Cdiscount: [Python](https://www.python.org/) (Scikit-learn) *vs.* [Spark](href="http://spark.apache.org/) (Mllib)


Il s'agit d'une version simplifiée du concours proposé par Cdiscount et paru sur le site [datascience.net](https://www.datascience.net/fr/challenge). Les données d'apprentissage sont accessibles sur demande auprès de Cdiscount mais les solutions de l'échantillon test du concours ne sont pas et ne seront pas rendues publiques. Un échantillon test est donc construit pour l'usage de ce tutoriel.  L'objectif est de prévoir la catégorie d'un produit à partir de son descriptif. Seule la catégorie principale (1er niveau, 47 classes) est prédite au lieu des trois niveaux demandés dans le concours. L'objectif est plutôt de comparer les performances des méthodes et technologies en fonction de la taille de la base d'apprentissage ainsi que d'illustrer sur un exemple complexe le prétraitement de données textuelles. 

Le jeux de données complet (15M produits) permet un test en vrai grandeur du **passage à l'échelle volume** des phases de préparation (*munging*), vectorisation (hashage, TF-IDF) et d'apprentissage en fonction de la technologie utilisée.

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

