# Ateliers: Technologies des Données Massives avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume**; processus qui transforme un statisticien en *Data Scientist*. 


# Recommandation de Films par Filtrage Collaboratif: [R](https://cran.r-project.org/) (softImpute) *vs.* [Spark](href="http://spark.apache.org/) (Mllib)


Les calepins traitent d'un problème classique de recommandation par filtrage collaboratif en comparant les ressources de la librairie [MLlib de Spark]([http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS) avec l'algorithme de complétion de grande matrice creuse implémenté dans la librairie [softImpute de R](https://cran.r-project.org/web/packages/softImpute/index.html). Le problème général est décrit dans une [vignette](http://wikistat.fr/pdf/st-atelier-recom-film.pdf) de [Wikistat](http://wikistat.fr/). Il est appliqué aux données publiques du site [GroupLens](http://grouplens.org/datasets/movielens/). L'objectif est donc de tester les méthodes et la procédure d'optimisation sur le plus petit jeu de données composé de 100k notes  de 943 clients sur 1682 films où chaque client a au moins noté 20 films. Le plus gros jeux de données  (20M notes) est utilisé pour **passer à l'échelle volume**. 

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

