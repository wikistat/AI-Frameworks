## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

## [Ateliers: Technologies des Données Massives](https://github.com/wikistat/Ateliers-Big-Data) avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume** des méthodes d'apprentissage; **processus qui transforme un statisticien en *Data Scientist*.** 


# Recommandation de Films par Filtrage Collaboratif: [R](https://cran.r-project.org/) ([`softImpute`](https://cran.r-project.org/web/packages/softImpute/index.html)) *vs.* [Spark](href="http://spark.apache.org/) ([`MLlib`](http://spark.apache.org/mllib/))

**Résumé** 
Les calepins traitent d'un problème classique de recommandation par filtrage collaboratif pour le commerce en ligne. L'objectif est de comparer les ressources de la librairie [MLlib de Spark]([http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS) avec l'algorithme de complétion de grande matrice creuse implémenté dans la librairie [`softImpute`](https://cran.r-project.org/web/packages/softImpute/index.html)  de R [(Mazmunder et al. 2010)](http://web.stanford.edu/~hastie/Papers/mazumder10a.pdf). Ces outils sont appliqués aux données publiques du site [GroupLens](http://grouplens.org/datasets/movielens/). L'objectif est donc de tester les méthodes et la procédure d'optimisation sur le plus petit jeu de données composé de 100k notes  de 943 clients sur 1682 films où chaque client a au moins noté 20 films. Le plus gros jeux de données  (20M notes) est utilisé pour **passer à l'échelle volume**. 

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

## Tutoriels
- [`Atelier-R-MovieLens`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/MovieLens/Atelier-R-MovieLens.ipynb): un calepin en R propose une comparaison élémentaire des méthodes de factorisation (SVD, NMF) et complétion de matrices sur un exemple "jouet" de dénombrements de ventes. La librairie [softImpute](https://web.stanford.edu/~hastie/swData/softImpute/vignette.html) de complétion de matrice est ensuite utilisée pour construire des recommandations à partir des données [MovieLens](http://grouplens.org/datasets/movielens/)
- [`Atelier-pyspark-MovieLens`](hhttps://github.com/wikistat/Ateliers-Big-Data/blob/master/MovieLens/Atelier-pyspark-MovieLens.ipynb):Le même objectif (recommandation MovieLens) est atteint dans un calepin en PySpark qui utilise la factorisation par [NMF de MLlib](http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html). 

**Attention**. La librairie [Scikit-learn](http://scikit-learn.org/stable/modules/decomposition.html#nmf) propose bien une version de NMF mais pas adpatée à l'objectif de complétion: les "0" de la matrices creuses sont des "0", pas des données manquantes. 

## Introduction aux [Systèmes de Recommandation](http://wikistat.fr/pdf/st-m-datSc3-colFil.pdf)
Les principes des systèmes de recommandation sont exposés plus en détail dans une [vignette](http://wikistat.fr/pdf/st-m-datSc3-colFil.pdf). En voici une brève présentation.

### Gestion de la relation client
La rapide expansion des sites de commerce en ligne a pour conséquence une explosion des besoins en marketing et *gestion de la relation client* (GRC ou CRM: *client relationship management*) spécifiques à ce type de média. C'est même le domaine qui est le principal fournisseur de données massives ou tout du moins la partie la plus visible de l'iceberg.

La GRC en marketing quantitatif traditionnel est principalement basée sur la construction de modèles de scores: d'appétence pour un produit, d'attrition (*churn*) ou risque de rompre un contrat. Voir à ce propos les scénarios d'appétence pour la [carte Visa Premier](https://github.com/wikistat/Apprentissage/tree/master/GRC-carte_Visa) ou celui concernant un produit d'[assurance vie](https://github.com/wikistat/Apprentissage/tree/master/Patrim-Insee) à partir de l'enquête INSEE sur le patrimoine des français.

### Commerce en ligne
Le commerce en ligne introduit de nouveaux enjeux avec la construction d'algorithmes de *filtrage* : sélection et recommandation automatique d'articles, appelés encore *systèmes de recommandation*. Certains concernent des méthodes *adaptatives* qui suivent la navigation de l'internaute, son flux de clics, jusqu'à l'achat ou non. Ces approches sont basées sur des algorithmes de bandits manchots et ne font pas l'objet de cet atelier. D'autres stratégies sont définies à partir d'un historique des comportements des clients, d'informations complémentaires sur leur profil, elles rejoignent les méthodes traditionnelles de marketing quantitatif. D'autres enfin sont basées sur la seule connaissance des interactions clients X produits à savoir la présence / absence d'achats ou un ensemble d'avis recueillis sous la forme de notes d'appréciation de chaque produit consommé. On parle alors de filtrage collaboratif (*collaborative filtering*).

### Filtrage collaboratif
Ce dernier cas a largement été popularisé par le concours [Netflix](http://www.netflixprize.com/) où il s'agit de proposer un film à un client en considérant seulement la matrice très creuse: clients X films, des notes sur une échelle de 1 à 5.  L'objectif est donc de prévoir le goût, la note, ou l'appétence d'un client pour un produit (livre, film...), qu'il n'a pas acheté, afin de lui proposer celui le plus susceptible de répondre à ses attentes. Tous les sites marchands: Amazon, Fnac, Netflix... implémentent de tels algorithmes.

Le filtrage collaboratif basé sur les seules interactions client X produits: présence / absence d'achat ou note d'appréciation  fait généralement appel à deux grandes familles de méthodes:

* *Méthode de voisinage* fondée sur des indices de similarité (corrélation linéaire ou des rangs de Spearman...) entre clients ou (exclusif) entre produits:

* *Modèle à facteurs* latents] basé sur une décomposition de faible rang avec une éventuelle contrainte de régularisation, de la matrice très creuse des notes ou avis clients X produits. 

La littérature est très abondante sur le sujet qui soulève plusieurs problèmes dont:

* comment évaluer un système de recommandation ? 
* Comment l'initier (*cold start problem*) ? C'est-à-dire comment initier une matrice très creuse avec très peu d'avis ou introduire de nouveaux clients ou produits.



Par ailleurs, des systèmes hybrides intègrent ces donnés d'interaction avec d'autres informations sur le profil des clients (variables âge, sexe, prénom...) ou encore sur  sur la typologie des produits (variables genre, année...).

### Modèles à facteurs latents
La recherche de facteurs latents est basée sur la [décomposition en valeurs singulière](http://wikistat.fr/pdf/st-m-explo-alglin.pdf) d'une matrice (SVD) ou la [factorisation d'une matrice non négative](http://wikistat.fr/pdf/st-m-explo-nmf.pdf) plus adaptée à des notes d'appréciation ou des effectifs de vente. Ces matrices sont généralement **très creuses**, souvent à peine 2% de valeurs connues sont renseignées et les autres, qui ne peuvent être mises à zéro, sont donc des valeurs *manquantes*.  il s'agit alors d'un problème de *complétion de matrice*.

## Données [MovieLens](http://grouplens.org/datasets/movielens/)
Des données réalistes croisant plusieurs milliers de clients et films, sont accessibles en ligne. Il s’agit d’une extraction du site [MovieLens](http://grouplens.org/datasets/movielens/) qui vous "aide" à choisir un film. Entre autres, quatre tailles de matrices creuses sont proposées:

- `100k` 100 000 évaluations de 1000 utilisateurs de 1700 films.
- `1M` Un million d’évaluations par 6000 utilisateurs sur 4000 films.
- `10M` Dix millions d’évaluation par 72 000 utilisateurs sur 10 000 fims.
- `20M` Vingt millions d’évaluations par 138 000 utilisateurs sur 27 000 films.

