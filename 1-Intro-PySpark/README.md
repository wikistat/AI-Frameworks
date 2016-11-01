###<a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

## [Ateliers: Technologies des Données Massives](https://github.com/wikistat/Ateliers-Big-Data) avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume** des méthodes d'apprentissage; **processus qui transforme un statisticien en *Data Scientist*.** 

**Remarques importantes**: si les données ne sont pas *grosses* à devoir être *distribuées*, un usage classique de Python voire R s'avère plus efficace pour une *phase d'aprpentissage*. En revanche, la phase de préparation des données (*data munging*), en flux ou pas, gagne à être opérée dans un environnement distribué (cf. [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099)). D'où l'importance pour le statisticien à se former à ces technologies.

# Introduction à [Spark](href="http://spark.apache.org/) avec ([`PySpark`](http://spark.apache.org/docs/latest/api/python/)) 



## Présentation de [Spark](href="http://spark.apache.org/)
Des données réellement massives sont systématiquement associées à une architecture distribuées de type *Hadoop*. Dans cet environnement spécifique, *Spark* occupe une place prépondérante. 

Techniquement, *Spark* manipule des **RDDs** (*resilient distributed datasets*) par des commandes en langage Java ou Scala mais il existe des API (*application programming interface*) acceptant des commandes en Python ([`PySpark`](http://spark.apache.org/docs/latest/api/python/)) et en  R. *Spark* intègre  beaucoup  de fonctionnalités réparties en quatre modules:

- [GraphX](http://spark.apache.org/graphx/) pour l’analyse de graphes ou réseaux, 
- [Streaming](http://spark.apache.org/streaming/) pour le traitement et l’analyse des flux,
- [SparkSQL](http://spark.apache.org/sql/) pour l’interrogation et la gestion de bases de tous types - 
- [MLlib et sparkML](http://spark.apache.org/mllib/) pour les principaux algorithmes d’apprentissage qui acceptent le passage à l'échelle (échelonnables ou *scalable*).

En plein développement, cet environnement comporte (version 1.6 2.0) des incohérences. SparkSQL génère et gère une nouvelle classe de données **DataFrame** (similaire à R) mais qui n’est pas connue de
*MLlib* qui va progressivement être remplacée par *SparkML* dans les versions à venir...


## Tutoriels d'nitiation à [Spark](href="http://spark.apache.org/) avec [`PySpark`](http://spark.apache.org/docs/latest/api/python/)
L'objectif de ces tutoriels est d'introduire les objets de la technologie [Spark](https://spark.apache.org/) et leur utilisation à l'aide de commandes en Python, plus précisément en utilisant l'API  [`PySpark`](http://spark.apache.org/docs/latest/api/python/). 

- [`Cal1-PySpark-munging`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/1-Intro-PySpark/Cal1-PySpark-munging.ipynb) justifie l'utilisation de cet environnement qui distribue automatiquement les données sur un cluster et parallélise les tâches; description des principaux types de données et du concept de *Resilient Distributed Datasets* (RDD): toute tâche en *Spark* s'exprime comme la création, la transformation de RDDs ou le lancement d'actions sur des RDDs. Ce sont les outils de préparation des données.

- [`Cal2-PySpark-statelem`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/1-Intro-PySpark/Cal2-PySpark-statelem.ipynb): Statistiques élémentaires et modélisaiton par régression logistique avec [MLlib](https://spark.apache.org/mllib/).
- [`Cal3-PySpark-SQL`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/1-Intro-PySpark/Cal3-PySpark-SQL.ipynb): Introduction à la gestion de *Data Frame* avec [SparkSQL](http://spark.apache.org/sql/). Requêtage, filtrage, exploratio élémentaire.

