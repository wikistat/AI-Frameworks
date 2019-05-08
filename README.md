## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Mathématiques Appliquées*](http://www.math.insa-toulouse.fr/fr/index.html), [`Science des Données`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

# Artificial Intelligence Frameworks

This course follows the [Machine Learning](https://github.com/wikistat/Apprentissage) and the [High Dimensional & Deep Learning](https://github.com/wikistat/High-Dimensional-Deep-Learning). In theses courses, you have acquired knowledge in machine and deep learning algoritms and their application on various type of data. This knowledge is primordial to become a DataScientist. 

As a DataScientist, you will also need to know the tool that wil allow you to perform these algorithms efficiently.
The two main goal of this course are : 
   * Discover this different tools:
      * Distributed computation with `Spark`.
      * Cloud computing with 'Google Cloud'.
      * Container with 'Docker`.
   * Use this tools on various domain of learning with real dataset and with usefull learning librairy.
      * Natural language processing with gensim
The main goal of this course is to have an overview of this tools and also mor algorithms...






<a href="https://www.python.org/"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Python_logo_and_wordmark.svg/390px-Python_logo_and_wordmark.svg.png" height="75" alt="Python"/></a> <a href="https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=emea-fr-all-en-dr-bkws-all-all-trial-e-gcp-1003963&utm_content=text-ad-none-any-DEV_c-CRE_167374210213-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20~%20EXA_1:1_FR_EN_General_Cloud_TOP_google%20cloud%20platform-KWID_43700016295756942-kwd-26415313501-userloc_9055236&utm_term=KW_google%20cloud%20platform-ST_google%20cloud%20platform&ds_rl=1242853&ds_rl=1245734&ds_rl=1245734&gclid=EAIaIQobChMIvaa_9OmL4gIVFeaaCh3jnQIfEAAYASAAEgJyp_D_BwE"><img src="https://cloud.google.com/_static/38e39c36bd/images/cloud/cloud-logo.svg" height="75" alt="Googe Cloud"/></a> 

 <a href="https://radimrehurek.com/gensim/"><img src="https://radimrehurek.com/gensim/_static/images/gensim.png" height="75" alt=Gensim/></a>    <a href="https://gym.openai.com/"><img src="https://gym.openai.com/assets/dist/home/header/home-icon-54c30e2345.svg" height="75" alt="Gym Open AI"/> </a> <a href="https://www.tensorflow.org/"><img src="https://avatars0.githubusercontent.com/u/15658638?s=200&v=4" height="75" alt="TensorFlow"/></a>   <a href="https://www.docker.com"><img src="https://www.docker.com/sites/default/files/social/docker_facebook_share.png" height="75" alt="Docker"/></a>    <a href="http://spark.apache.org/"><img src="https://spark.apache.org/images/spark-logo-trademark.png" height="75" alt="Spark"/> </a> 

## Knowledge requirements

- [R Tutorial](https://github.com/wikistat/Intro-R)
- [Python Tutorial](https://github.com/wikistat/Intro-Python)
- [Elementary statistic tools](https://github.com/wikistat/StatElem)
- [Data Exploration and Clustering](https://github.com/wikistat/Exploration). 
- [Machine Learning](https://github.com/wikistat/Apprentissage)
- [High Dimensional & Deep Learning](https://github.com/wikistat/High-Dimensional-Deep-Learning)


## Schedule

The course is divided in 5 sessions.

- **Session 1** *Spark*: [Complément](https://github.com/wikistat/Intro-Python/blob/master/Cal4-PythonProg.ipynb) de programmation en Python puis [Initiation](https://github.com/wikistat/Intro-PySpark) à Spark *via* l'API `PySpark`.
- **Session 2** *Reconnaissance d'images*; apprentissage profond avec `TensorFlow` utilisé *via* `Keras`. Deux cas d'usage: [`MNIST`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/MNIST) et [cats *vs.* dogs](https://github.com/wikistat/Ateliers-Big-Data/tree/master/CatsVSDogs) avec transfert de l'apprentissage du  réseau `inception` de Google. 
- **Session 3** *en chantier* [*Cloud computing*]() accéder à *Google Cloud* pour passer à l'échelle; transférer un code de calcul, l'exécuter.
- **Session 4** *MLlib*: pratique de la librairie MLlib de Spark pour l'apprentissage sur données distribuées. Comparaison des performances en R, Python et MlLib sur des cas d'usage
   - [`MNIST`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/MNIST) reconnaissance de caractères manuscrits avec R, `Scikit-learn` et MLlib. Laisser la version avec `keras` est traité dans l'épisode 2.
   - [`MovieLens`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/MovieLens) Système de recommandation de films avec R (`softImpute`) et MLlib (*non negative matrix factorisation*).
- **Session 5** *NLP (Natural Language Processing)*
   - *sentiment analysis*
   - [`Cdiscount`](https://github.com/wikistat/Ateliers-Big-Data/tree/master/Cdiscount) catégorisation de produits (*text mining*): Scikit-learn *vs.* MLlib.
   - *word embedding*

Une synthèse des résultats obtenus dans les 3 cas d'usage (MNIST, MovieLens, Cdiscount) est développée par [Besse et al. 2016](https://hal.archives-ouvertes.fr/hal-01350099).

Chaque année, l'**évaluation** des compétences acquises est basée sur la participation (résultats et soutenance orale) des étudiants à un [défi IA](https://defi-ia.insa-toulouse.fr/) dont l'objet est la construction d'une meilleure prévision par apprentissage sur un jeu complexe de données.



