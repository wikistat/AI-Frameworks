## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

# Science des Données & Statistique

### [Lire plus...](http://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-lm-Intro-Stat_SD.pdf)

Schématiquement, la **Science des Données** est définie autour d'une *agrégation de compétences* en Informatique (langage comme [R](href="https://cran.r-project.org/) et [Python](https://www.python.org/) , gestion des données, calcul parallèle...), Statistique (exploration, estimation test, modélisation, prévision) Apprentissage Machine (prévision), Mathématiques (probabilités, optimisation, analyse fonctionnelle, graphes...). 

Son **apprentissage** est acquis par l'intermédiaire de scénarios d'analyse de données réelles, ou *tutoriel*, présentés sous forme de *calepins* ([*jupyter notebooks*](http://jupyter.org/)) en [R](href="https://cran.r-project.org/) ou [Python](https://www.python.org/). Voir à ce sujet le [livre de référence](https://www.inferentialthinking.com/) du cours [*Fondations of Data Science*](http://data8.org/) de l'UC Berkley.

Cette **pratique** est **indispensable** mais masque les *aspects théoriques* (mathématiques, statistiques): une *formule* est remplacée par un commande ou fonction en Python ou R, une *démonstration* par l'exécution d'exemples dans un calepin.

Pour offrir de la *profondeur*, plus de compréhension, à cette (auto)-formation, les calepins renvoient (liens hypertextes) systématiquement à des **vignettes "théoriques"**  du site [wikistat.fr](http://wikistat.fr/) exposant en détail (cours) les méthodes et algorithmes concernés.

Il ne s'agit pas simplement de pouvoir exécuter une méthode, un algorithme, il est important d'en **comprendre les propriétés**, conditions d'utilisation et limites.

# Saison 4 [*Technologies des Grosses Data*](https://github.com/wikistat/Ateliers-Big-Data) 

### [Introduction plus détaillée](http://www.math.univ-toulouse.fr/~besse/Wikistat/pdf/st-m-Intro-AtelierGD.pdf)

Si les données ne sont pas *grosses* à devoir être *distribuées*, un usage classique de Python voire R s'avère plus efficace pour une *phase d'apprentissage*. En revanche, la phase de préparation de données massives (*data munging*), en flux ou pas, gagne beaucoup à être opérée dans un environnement distribué (*Hadoop*) utilisant *Spark*, notamment *via* l'API `PySpark` (cf. [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099)). 

Dans tous les cas, l'apprentissage de modèles complexes (*deep learning* et `keras`, `XGBoost`...) nécessite des moyens de calcul massivement parallèles (*e.g. GPU, cloud*). D'où l'importance pour le statisticien / *data scientist* à se former aux technologies afférentes.

## Objectifs

Cette saison est consacrée au **passage à l'échelle** pour l'analyse de *grosses* *data*, donc distribuées sur plusieurs machines (*Hadoop*) ou de données en grande dimension. L'objectif est de comparer les performances des principaux environnements ou architectures plus ou moins bien adaptées à l'analyse de données massives en fonction du but visé: préparation des donnés, exploration, apprentissage, prévision.

## Prérequis
Avoir acquis les compétences des saisons précédentes ou y revenir:

- [Initiation à R](https://github.com/wikistat/Intro-R)
- [Initiation à Python](https://github.com/wikistat/Intro-Python)
- Formation aux [outils Statistiques de base](https://github.com/wikistat/StatElem)
- [Exploration Statistique pour la Science des Données](https://github.com/wikistat/Exploration). Cete saison intègre les algorithmes d'apprentissage non-supervisé (*clustering*).
- [Apprentissage Machine / Statistique](https://github.com/wikistat/Apprentissage)

## Épisodes ou cas d'usage

- Épisode 1 [`Intro-Spark`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/1-Intro-PySpark) Tutoriels d'initiation à Spark en Python (PySpark)
- Épisode 2 [`MNIST`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/2-MNIST) reconnaissance de caractères manuscrits, 
- Épisode 3 [`MovieLens`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/3-MovieLens) recommandation de films 
- Épisode 4 [`Cdiscount`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/4-Cdiscount) catégorisation de produits (*text mining*).

**En travaux:**

- Épisode 5 [`Human Activity Recognition`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/5-HumanActivityRecognition) (HAR) et utilisation de `XGBoost`, LSTM avec`Keras` (*TensorFlow*).
- Episode 6 [`Cats vs. Dogs`]() apprentissage profond (réseau convolutionnel) avec `Keras` (*TensorFlow*).
- Episode 7 [`Curves anomalies`]()

Une synthèse des résultats obtenus dnas les 3 cas d'usage (MNIST, MovieLens, Cdiscount) est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

Chaque année, l'**évaluation** des compétences acquises est basée sur la participation (résultats et soutenance orale) des étudiants à un [défi grosses data](https://defibigdata2018.insa-toulouse.fr/) dont l'objet est la construction d'une meilleure prévision par apprentissage sur un jeu complexe de données.

## De [Hadoop](http://hadoop.apache.org/) à [Spark](http://spark.apache.org/)

### Introduction à [Hadoop](http://hadoop.apache.org/)

Hadoop (*distributed file system*) est devenu la technologie systématiquement associée à la notion de données considérées comme massives car distribuées. Largement développé par Google
avant de devenir un projet de la fondation *Apache*, cette technologie répond à des besoins spécifiques de centres de données: stocker
des volumétries considérables en empilant des milliers de cartes d’ordinateurs et disques de faible coût, plutôt que de mettre en œuvre un supercalculateur, tout en préservant la fiabilité par une forte tolérance aux pannes. Les données sont dupliquées et, en cas de défaillance, le traitement est poursuivi, une carte remplacée,
les données automatiquement reconstruites, sans avoir à arrêter le système. Ce type d’architecture génère en revanche un coût algorithmique. Les nœuds de ces serveurs ne peuvent communiquer que par couples (clef, valeur) et les différentes étapes d’un traitement doivent pouvoir être décomposées en étapes fonctionnelles élémentaires comme celles dites de *MapReduce*

Pour des opérations simples, par exemple de dénombrement de mots, d’adresses URL, des statistiques élémentaires, cette architecture s’avère efficace. Une étape *Map* réalise, en parallèle, les dénombrements à chaque nœud ou exécuteur (*workers*) d’un *cluster*
d’ordinateurs, le résultat est un ensemble de couples : une clef (le mot, l’URL à dénombrer) associée à un résultat partiel. Les clefs identiques sont regroupées dans une étape intermédiaire de tri (*shuffle*) au sein d’une même étape *Reduce* qui fournit pour chaque clef le résultat final.

Dans cette architecture, les algorithmes sont dits *échelonnables* de l’anglais **scalable** si  le  temps  d’exécution  décroît  linéairement  avec  le  nombre  d’exécuteurs dédiés au calcul. C’est immédiat pour des dénombrements, des calculs de
moyennes, ce n’est pas nécessairement le cas pour des algorithmes itératifs complexes. 

### Pourquoi [Spark](http://spark.apache.org/)?
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


## Mise en oeuvre de [Spark](http://spark.apache.org/)
 
Un premier groupe (dossier [1-Intro-Spark](https://github.com/wikistat/Ateliers-Big-Data/tree/master/1-Intro-PySpark)) de tutoriels, propose une initiation à l'utilisation en Python de l'environnement (*framework*) [*Spark*](http://spark.apache.org/) devenu une référence pour la gestion et l'analyse de données distribuées (*e.g* sous *Hadoop*). L'accent est mis sur la gestion des RDDs (*resilient distributed datasets*) et leur analyse à l'aide des librairies *Mllib* et *SparkML*.



