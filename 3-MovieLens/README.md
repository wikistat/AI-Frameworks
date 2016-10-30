# [Ateliers: Technologies des Données Massives](https://github.com/wikistat/Ateliers-Big-Data) avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume** des méthodes d'apprentissage; **processus qui transforme un statisticien en *Data Scientist*.** 


# Recommandation de Films par Filtrage Collaboratif: [R](https://cran.r-project.org/) (softImpute) *vs.* [Spark](href="http://spark.apache.org/) (Mllib)

**Résumé** 
Les calepins traitent d'un problème classique de recommandation par filtrage collaboratif pour le commerce en ligne. L'objectif est de comparer les ressources de la librairie [MLlib de Spark]([http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS) avec l'algorithme de complétionl de grande matrice creuse implémenté dans la librairie [softImpute de R](https://cran.r-project.org/web/packages/softImpute/index.html). Ces outils sont appliqués aux données publiques du site [GroupLens](http://grouplens.org/datasets/movielens/). L'objectif est donc de tester les méthodes et la procédure d'optimisation sur le plus petit jeu de données composé de 100k notes  de 943 clients sur 1682 films où chaque client a au moins noté 20 films. Le plus gros jeux de données  (20M notes) est utilisé pour **passer à l'échelle volume**. 

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

## 1 Systèmes de Recommandation
Les principes des systèmes de recommandation sont exposés en détail dans une [vignette](). En voici une brève présentation.

### 1.1 Gestion de la relation client
La rapide expansion des sites de commerce en ligne a pour conséquence une explosion des besoins en marketing et *gestion de la relation client* (GRC ou CRM: *client relationship management*) spécifiques à ce type de média. C'est même le domaine qui est le principal fournisseur de données massives ou tout du moins la partie la plus visible de l'iceberg.

La GRC en marketing quantitatif traditionnel est principalement basée sur la construction de modèles de scores: d'appétence pour un produit, d'attrition (*churn*) ou risque de rompre un contrat. Voir à ce propos les scénarios d'appétence pour la [carte Visa Premier](http://wikistat.fr/pdf/st-scenar-app-visa.pdf) ou celui concernant un produit d'[assurance vie](http://wikistat.fr/pdf/st-scenar-app-patrimoine.pdf) à partir de l'enquête INSEE sur le patrimoine des français.

### 1.2 Commerce en ligne
Le commerce en ligne introduit de nouveaux enjeux avec la construction d'algorithmes de *filtrage* : sélection et recommandation automatique d'articles, appelés encore *systèmes de recommandation*. Certains concernent des méthodes *adaptatives* qui suivent la navigation de l'internaute, son flux de clics, jusqu'à l'achat ou non. Ces approches sont basées sur des algorithmes de bandits manchots et ne font pas l'objet de cet atelier. D'autres stratégies sont définies à partir d'un historique des comportements des clients, d'informations complémentaires sur leur profil, elles rejoignent les méthodes traditionnelles de marketing quantitatif. D'autres enfin sont basées sur la seule connaissance des interactions clients X produits à savoir la présence / absence d'achats ou un ensemble d'avis recueillis sous la forme de notes d'appréciation de chaque produit consommé. On parle alors de filtrage collaboratif (*collaborative filtering*).

### 1.3 Filtrage collaboratif
Ce dernier cas a largement été popularisé par le concours [Netflix](http://www.netflixprize.com/) où il s'agit de proposer un film à un client en considérant seulement la matrice très creuse: clients X films, des notes sur une échelle de 1 à 5.  L'objectif est donc de prévoir le goût, la note, ou l'appétence d'un client pour un produit (livre, film...), qu'il n'a pas acheté, afin de lui proposer celui le plus susceptible de répondre à ses attentes. Tous les sites marchands: Amazon, Fnac, Netflix... implémentent de tels algorithmes.

Le filtrage collaboratif basé sur les seules interactions client X produits: présence / absence d'achat ou note d'appréciation  fait généralement appel à deux grandes familles de méthodes:

* *Méthode de voisinage* fondée sur des indices de similarité (corrélation linéaire ou des rangs de Spearman...) entre clients ou (exclusif) entre produits:

* *Modèle à facteurs* latents] basé sur une décomposition de faible rang avec une éventuelle contrainte de régularisation, de la matrice très creuse des notes ou avis clients X produits. 

La littérature est très abondante sur le sujet qui soulève plusieurs problèmes dont:

* comment évaluer un système de recommandation ? 
* Comment l'initier (*cold start problem*) ? C'est-à-dire comment initier une matrice très creuse avec très peu d'avis ou introduire de nouveaux clients ou produits.



Par ailleurs, Des systèmes hybrides intègrent ces donnés d'interaction avec d'autres informations sur le profil des clients (variables âge, sexe, prénom...) ou encore sur  sur la typologie des produits (variables genre, année...).

\subsection{Modèles à facteurs latents}
La recherche de facteurs latents est basée sur la \href{http://wikistat.fr/pdf/st-m-explo-alglin.pdf}{décomposition en valeurs singulière} d'une matrice (SVD) ou la \href{http://wikistat.fr/pdf/st-m-explo-nmf.pdf}{factorisation d'une matrice non négative} plus adaptée à des notes d'appréciation. Ces matrices sont généralement très creuses, souvent à peine 2\% de valeurs connues sont renseignées et les autres qui ne peuvent être mises à zéro, sont donc des valeurs \emph{manquantes}.  il s'agit alors d'un problème de complétion de matrice\footnote{Le \href{https://sites.google.com/site/igorcarron2/matrixfactorizations}{site d'Igor Carron} donne un aperçu assez réaliste de la complexité et du foisonnement de la recherche sur ce thème.}.

Le scénario de présentation des méthodes de \href{http://wikistat.fr/pdf/st-scenar-explo7-nmf.pdf}{factorisation de matrices (SVD, NMF) et complétion} introduit ce type d'approche sur un exemple "jouet" à partir d'une matrice de dénombrements de ventes. La prise en compte d'une matrice de notes avec des "0" correspondant à des données manquantes limite les possibilités.