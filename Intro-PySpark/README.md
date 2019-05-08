###<a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

# <a href="http://spark.apache.org/"><img src="https://www.versive.com/user/pages/01.home/_results/spark.svg?g-7518fe06" style="max-width: 100px; display: inline" alt="Spark"/> </a> [pour Statistique et *Science des* grosses *Données*](https://github.com/wikistat/Intro-Python)

L'objectifs de ces tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire **Spark** comme outil privilégié pour *passage à l'échelle Volume* des méthodes d'apprentissage; processus qui transforme un statisticien en *Data Scientist*.

**Remarques importantes**: si les données ne sont pas *grosses* à devoir être *distribuées*, un usage classique de Python voire R s'avère plus efficace pour une *phase d'apprentissage*. En revanche, la phase de préparation des données (*data munging*), en flux ou pas, gagne à être opérée dans un environnement distribué (cf. [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099)). D'où l'importance pour le statisticien à se former à ces technologies.


## Tutoriels d'initiation à Spark  avec [`PySpark`](http://spark.apache.org/docs/latest/api/python/)
L'objectif de ces tutoriels est d'introduire les objets de la technologie [Spark](https://spark.apache.org/) et leur utilisation à l'aide de commandes en Python, plus précisément en utilisant l'API  [`PySpark`](http://spark.apache.org/docs/latest/api/python/). 

- [`Cal1-PySpark-munging`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/Intro-PySpark/Cal1-PySpark-munging.ipynb) justifie l'utilisation de cet environnement qui distribue automatiquement les données sur un cluster et parallélise les tâches; description des principaux types de données et des concepts de *Resilient Distributed Datasets* (RDD) et *DataFrame*.
- [`Cal2-PySpark-statelem`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/Intro-PySpark/Cal2-PySpark-statelem.ipynb): Statistiques élémentaires et modélisation par régression logistique avec [MLlib](https://spark.apache.org/mllib/).
- [`Cal3-PySpark-SQL`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/Intro-PySpark/Cal3-PySpark-SQL.ipynb): Introduction à la gestion de *Data Frame* avec [SparkSQL](http://spark.apache.org/sql/). Requêtage, filtrage, exploration élémentaire.

- [`Cal4-PySpark-Statelem&Pipeline-SparkML`](https://github.com/wikistat/Ateliers-Big-Data/blob/master/Intro-PySpark/Cal4-PySpark-Statelem&Pipeline-SparkML.ipynb): Statistiques élémentares et définition de pipelines afin d'enchaîner les traitements.



## De [Hadoop](http://hadoop.apache.org/) à Spark

### Introduction à [Hadoop](http://hadoop.apache.org/)

Hadoop (*distributed file system*) est devenu la technologie systématiquement associée à la notion de données considérées comme massives car distribuées. Largement développé par Google
avant de devenir un projet de la fondation *Apache*, cette technologie répond à des besoins spécifiques de centres de données: stocker
des volumétries considérables en empilant des milliers de cartes d’ordinateurs et disques de faible coût, plutôt que de mettre en œuvre un supercalculateur, tout en préservant la fiabilité par une forte tolérance aux pannes. Les données sont dupliquées et, en cas de défaillance, le traitement est poursuivi, une carte remplacée,
les données automatiquement reconstruites, sans avoir à arrêter le système. Ce type d’architecture génère en revanche un coût algorithmique. Les nœuds de ces serveurs ne peuvent communiquer que par couples (clef, valeur) et les différentes étapes d’un traitement doivent pouvoir être décomposées en étapes fonctionnelles élémentaires comme celles dites de *MapReduce*

Pour des opérations simples, par exemple de dénombrement de mots, d’adresses URL, des statistiques élémentaires, cette architecture s’avère efficace. Une étape *Map* réalise, en parallèle, les dénombrements à chaque nœud ou exécuteur (*workers*) d’un *cluster*
d’ordinateurs, le résultat est un ensemble de couples : une clef (le mot, l’URL à dénombrer) associée à un résultat partiel. Les clefs identiques sont regroupées dans une étape intermédiaire de tri (*shuffle*) au sein d’une même étape *Reduce* qui fournit pour chaque clef le résultat final.

Dans cette architecture, les algorithmes sont dits *échelonnables* de l’anglais **scalable** si  le  temps  d’exécution  décroît  linéairement  avec  le  nombre  d’exécuteurs dédiés au calcul. C’est immédiat pour des dénombrements, des calculs de
moyennes, ce n’est pas nécessairement le cas pour des algorithmes itératifs complexes. 

### Pourquoi Spark?
Les algorihtmes de certains méthodes s'adaptent facilement aux contraintes de *MapReduce* d'autres pas et cela opère une *sélection naturelle* des méthodes qui passent facilement à l'échelle volume. Anisi, la méthode des *k-plus proches voisins* n’est pas échelonnable au contraire des algorithmes de classification non-supervisée par réallocation dynamique (e.g.Forgy, *k-means*) qui peuvent opérer par itérations d’étapes *MapReduce*.

Mais, même *scalable* ou *échelonnable*, les méthodes itératives soulèvent d'autres problèmes. L’exemple de l’algorithme de Forgy (1965) est très révélateur.

- **Initialisation** de l’algorithme par définition d’une fonction de distance et désignation aléatoire de *k centres*
- **Jusqu’à convergence**, itérer
 - L’étape **Map** calcule, en parallèle, les distances de chaque observation aux *k* centres courants. Chaque observation (vecteur
de  valeurs)  est  affectée  au  centre  (clef)  le  plus  proche.  Les couples: (*clef* ou numéro de centre, vecteur des *valeurs*) sont
communiqués à l’étape *Reduce*.
 - Une étape intermédiaire implicite (**Shuffle**) adresse les couples
de même *clef* à la même étape *Reduce* suivante.
 - Pour chaque *clef* désignant un groupe, une étape **Reduce** calcule
les nouveaux barycentres, moyennes des valeurs des variables des individus partageant la même classe, c’est-à-dire la même valeur de *clef*.

**Problème** de cette implémentation, le temps d’exécution économisé par la parallélisation des calculs est fortement pénalisé par la nécessité d’écrire et relire toutes les données entre deux itérations.

C'est une des principales motivations la mise en place de la technologie **Spark** (Zaharia et al.[2012]). L'autre est la **capacité** de cet environnement à lire, gérer, tout type de fichier ou d'architecture de données distribuées ou pas.

Cette couche logicielle au-dessus de systèmes de gestion de fichiers comme Hadoop introduit la notion de **base de données résiliente** (*resilient distributed dataset* ou **RDD**) dont chaque partition reste, si nécessaire, présente en mémoire entre deux itérations pour éviter réécriture et relecture. Cela répond bien aux principales contraintes: *des données massives ne doivent pas être déplacées* et un résultat doit être obtenu par *une seule opération de lecture sur disque*.


### Introduction à Spark

Techniquement, *Spark* manipule des **RDDs** (*resilient distributed datasets*) par des commandes en langage Java ou Scala mais il existe des API (*application programming interface*) acceptant des commandes en Python ([`PySpark`](http://spark.apache.org/docs/latest/api/python/)) et en  R. *Spark* intègre  beaucoup  de fonctionnalités réparties en quatre modules:

- [GraphX](http://spark.apache.org/graphx/) pour l’analyse de graphes ou réseaux, 
- [Streaming](http://spark.apache.org/streaming/) pour le traitement et l’analyse des flux,
- [SparkSQL](http://spark.apache.org/sql/) pour l’interrogation et la gestion de bases de tous types - 
- [MLlib et sparkML](http://spark.apache.org/mllib/) pour les principaux algorithmes d’apprentissage qui acceptent le passage à l'échelle (échelonnables ou *scalable*).

En plein développement, cet environnement comporte (version 2.3) des incohérences. SparkSQL génère et gère une nouvelle classe de données ***DataFrame***, par références aux classes de R et pandas, et qui est progressivement prise en charge dans les autres modules. *MLlib* fut en partie remplacée par *SparkML* pour intégrer la classe *DataFrame* aux algortrithmes d'apprentissage mais redevient finalement *MLlib* avec cette fonctionnalité. Cette migration devrait être achevée acvec la verions 3.0 de Spark.


