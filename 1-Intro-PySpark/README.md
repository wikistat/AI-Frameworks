# Ateliers: Technologies des Données Massives avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume**; processus qui transforme un statisticien en *Data Scientist*. 


## Initiation à [Spark](href="http://spark.apache.org/) en PySpark


Données réellement massives sont systématiquement associées à une architecture distribuées de type *Hadoop* dans cet environnement spécifique, Spark prend une place de plus en plus prépondérante. 

L'objectif de ces tutoriels est d'introduire les objets de la technologie [Spark](https://spark.apache.org/) et leur utilisation à l'aide de commandes en Python, plus précisément en utilisant l'API  [`PySpark`](http://spark.apache.org/docs/latest/api/python/). 

Le premier tutoriel justifie l'utilisation de cet environnement qui distribue automatiquement les données sur un cluster et parallélise les tâches; description des principaux types de données et du concept de *Resilient Distributed Datasets* (RDD): toute tâche en *Spark* s'exprime comme la création, la transformation de RDDs ou le lancement d'actions sur des RDDs.

Le deuxième aborde les traitements statistiques élémentaires, la gestion de *data frame*, les premières modélisations. L'utilisation plus élaborée des ressources des librairies SparkML et MLlib pour l'apprentissage statistique est développée dans les autres ateliers. 
