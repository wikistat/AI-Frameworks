# [Ateliers: Technologies des Données Massives](https://github.com/wikistat/Ateliers-Big-Data) avec [R](https://cran.r-project.org/), [Python](https://www.python.org/) et / ou [Spark](href="http://spark.apache.org/)

L'objectifs de ces ateliers ou tutoriels sous forme de calepins ([*jupyter notebooks*](http://jupyter.org/)) est d'introduire le **passage à l'échelle Volume** des méthodes d'apprentissage; **processus qui transforme un statisticien en *Data Scientist*.** 

**Remarques importantes**: si les données ne sont pas *grosses* à devoir être *distribuées*, un usage classique de Python voire R s'avère plus efficace pour une *phase d'aprpentissage*. En revanche, la phase de préparation des données (*data munging*), en flux ou pas, gagne à être opérée dans un environnement distribué (cf. [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099)). D'où l'importance pour le statisticien à se former à ces technologies.


## De [Hadoop](http://hadoop.apache.org/) à [Spark](http://spark.apache.org/)

Un [rapport](https://hal.archives-ouvertes.fr/file/index/docid/995801/filename/st-stat-bigdata-clones.pdf) introduit aux technologies adaptées à des données massives donc *distribuées*, principalement à l'environnement **Hadoop** et aux fonctionnalités *MapReduce*.

### Introduction à [Hadoop](http://hadoop.apache.org/)

Hadoop (*distributed file system*) est devenu la technologie systématiquement associée à la notion de données considérées comme massives car distribuées. Largement développé par Google
avant de devenir un projet de la fondation *Apache*, cette technologie répond à des besoins spécifiques de centres de données: stocker
des volumétries considérables en empilant des milliers de cartes d’ordinateurs et disques de faible coût, plutôt que de mettre en œuvre un supercalculateur, tout en préservant la fiabilité par une forte tolérance aux pannes. Les données sont dupliquées et, en cas de défaillance, le traitement est poursuivi, une carte remplacée,
les données automatiquement reconstruites, sans avoir à arrêter le système. Ce type d’architecture génère en revanche un coût algorithmique. Les nœuds de ces serveurs ne peuvent communiquer que par couples (clef, valeur) et les différentes étapes d’un traitement doivent pouvoir être décomposées en étapes fonctionnelles élémentaires comme celles dites de *MapReduce*

Pour des opérations simples, par exemple de dénombrement de mots, d’adresses URL, des statistiques élémentaires, cette architecture s’avère efficace. Une étape *Map* réalise, en parallèle, les dénombrements à chaque nœud ou exécuteur (*workers*) d’un *cluster*
d’ordinateurs, le résultat est un ensemble de couples : une clef (le mot, l’URL à dénombrer) associée à un résultat partiel. Les clefs identiques sont regroupées dans une étape intermédiaire de tri (*shuffle*) au sein d’une même étape *Reduce* qui fournit pour chaque clef le résultat final.

Dans cette architecture, les algorithmes sont dits *échelonnables* de l’anglais **scalable** si  le  temps  d’exécution  décroît  linéairement  avec  le  nombre  d’exécuteurs dédiés au calcul. C’est immédiat pour des dénombrements, des calculs de
moyennes, ce n’est pas nécessairement le cas pour des algorithmes itératifs complexes. 

### Pourquoi [Spark](http://spark.apache.org/)
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

**Principal problème** de cette implémentation, le temps d’exécution économisé par la parallélisation des calculs est fortement pénalisé par la nécessité d’écrire et relire toutes les données entre deux itérations.

C'est une des principales motivations la mise en place de la technologie **Spark** (Zaharia et al.[2012]). L'autre est la capacité de cet environnement à lire, gérer, tout type de fichier ou d'architecture de données distribuées ou pas.

Cette couche logicielle au-dessus de systèmes de gestion de fichiers comme Hadoop introduit la notion de **base de données résiliente** (*
resilient distributed dataset* ou **RDD**) dont chaque partition reste, si nécessaire, présente en mémoire entre deux itérations pour éviter réécriture et relecture. Cela répond bien aux principales contraintes: *des données massives ne doivent pas être déplacées* et un résultat doit être obtenu par *une seule opération de lecture sur disque*.


## Mise en oeuvre de [Spark](http://spark.apache.org/)
 
Un premier groupe (dossier [1-Intro-Spark](https://github.com/wikistat/Ateliers-Big-Data/tree/master/1-Intro-PySpark)) de tutoriels, propose une initiation à l'utilisation en Python de l'environnement (*framework*) [*Spark*](http://spark.apache.org/) devenu une référence pour la gestion et l'analyse de données distribuées (*e.g* sous *Hadoop*). L'accent est mis sur la gestion des RDDs (*resilient distributed datasets*) et leur analyse à l'aide des librairies *Mllib* et *SparkML*.

## [Cas d'usage](https://hal.archives-ouvertes.fr/hal-01350099)
Chacun des autres dossiers concerne un jeu de données et contient un ou des calepins (*notebooks*) au format .ipynb codés en R, Python ou PySpark à télécharger et ouvrir dans *Jupyter*. L'objectif est de comparer les performances des principaux environnements plus ou moins bien adaptés à l'analyse de données massives en fonction du but visé.

Il est question de [reconnaissance de caractères](http://localhost:8888/tree/Ateliers-Big-Data/2-MNIST) (MNIST), de [recommandation de films](http://localhost:8888/tree/Ateliers-Big-Data/3-MovieLens) (MovieLens) et de [catégorisation de produits](http://localhost:8888/tree/Ateliers-Big-Data/4-Cdiscount) (*text mining*).

La synthèse des résultats obtenus est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

