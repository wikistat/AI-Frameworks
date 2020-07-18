## <a href="http://www.insa-toulouse.fr/" ><img src="http://www.math.univ-toulouse.fr/~besse/Wikistat/Images/Logo_INSAvilletoulouse-RVB.png" style="float:left; max-width: 80px; display: inline" alt="INSA"/> |  [*Applied mathematics*](http://www.math.insa-toulouse.fr/fr/index.html), [`Data Science`](http://www.math.insa-toulouse.fr/fr/enseignement.html) 

# Artificial Intelligence Frameworks

This course follows the [Machine Learning](https://github.com/wikistat/Apprentissage) and the [High Dimensional & Deep Learning](https://github.com/wikistat/High-Dimensional-Deep-Learning) courses.
 In theses courses, you have acquired knowledge in machine and deep learning algoritms and their application on various type of data. This knowledge is primordial to become a DataScientist. 

This course has three main objectives. You will 

1. learn how to apply efficiently these algorithms using
    * Cloud computing with [Google Cloud](https://cloud.google.com/gcp/?utm_source=google&utm_medium=cpc&utm_campaign=emea-fr-all-en-dr-bkws-all-all-trial-e-gcp-1003963&utm_content=text-ad-none-any-DEV_c-CRE_167374210213-ADGP_Hybrid%20%7C%20AW%20SEM%20%7C%20BKWS%20~%20EXA_1:1_FR_EN_General_Cloud_TOP_google%20cloud%20platform-KWID_43700016295756942-kwd-26415313501-userloc_9055236&utm_term=KW_google%20cloud%20platform-ST_google%20cloud%20platform&ds_rl=1242853&ds_rl=1245734&ds_rl=1245734&gclid=EAIaIQobChMIvaa_9OmL4gIVFeaaCh3jnQIfEAAYASAAEgJyp_D_BwE),
    * Container with [Docker](https://www.docker.com),

2. discover new field of artificial intelligence applied on (real) datasets that require specific algorithms:
    * **Natural language processing**.
        * <ins>Algorithms</ins>: Text processing, Vectorizer, Words Embedding, RNN
        * *Libraries :  [Nltk](https://www.nltk.org/), [Scikit-Learn](https://www.tensorflow.org/), [Gensim](https://gym.openai.com/)*
    * **Video Game**
        * <ins>Algorithms</ins>: Reinforcement learning, (Policy Gradient algorithm, Q-Learning, Deep Q-learning)
        * *Libraries :  [AI Gym](https://gym.openai.com/), [Tensorflow](https://www.tensorflow.org/).
    * **Movies Notations**
        * <ins>Algorithms</ins>: Recommendation system, (User/User and Item/Item filters, NMF, Neural recomendation system)
        * *Libraries :  [Surprise](https://surprise.readthedocs.io/en/stable/index.html), [Tensorflow](https://www.tensorflow.org/).

3. how to efficiently share reproducible code. 
    * Build a Github repository.

**NB**: Some contents from previous years are still available on the repository (like **Spark**) but are not covered during theses courses anymore. 


## Knowledge requirements

- [R Tutorial](https://github.com/wikistat/Intro-R)
- [Python Tutorial](https://github.com/wikistat/Intro-Python)
- [Elementary statistic tools](https://github.com/wikistat/StatElem)
- [Data Exploration and Clustering](https://github.com/wikistat/Exploration). 
- [Machine Learning](https://github.com/wikistat/Apprentissage)
- [High Dimensional & Deep Learning](https://github.com/wikistat/High-Dimensional-Deep-Learning)


## Schedule

* Lectures : 10 hours
* Practical Works : 30 hours.

The course is divided in 5 sessions.

Course introduction: [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/Course%20Introduction.pdf) <br>
Github Reminder: [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/Github%20-%20Reminder.pdf)


- **Session 1 - 02-11-20**
   - Natural language processing: Text Cleaning + Text Vectorization
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Frameworks_NLP_RNN.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/blob/master/NatualLangageProcessing/1_cleaning_vectorization.ipynb)
   - Natural language processing: Words Embedding.
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Frameworks_NLP_RNN.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/blob/master/NatualLangageProcessing/2_words_embedding.ipynb)
- **Session 2 - 16-11-20**
   - Natural language processing: Recurrent Network
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Frameworks_NLP_RNN.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/blob/master/NatualLangageProcessing/3_recurrent_neural_network.ipynb)
   - Python environment + Github Repo + Python Script.
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/Python%20-%20Script.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/tree/master/CloudComputing)
- **Session 3 - 30-11-20**
   - Introduction to Google Cloud Computing.
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/Google%20cloud.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/tree/master/CloudComputing)
   - Docker
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Frameworks_GCE.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/tree/master/CloudComputing)
- **Session 4 - 07-12-20**
   - Reinforcement learning: PG Gradient
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Frameworks_RL.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/tree/master/ReinforcementLearning)
   - Reinforcement learning: Deep Q-learning
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Frameworks_RL.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/tree/master/ReinforcementLearning)
- **Session 5 14-12-20**
   - Recommendation System. 
        * [Slides](https://github.com/wikistat/AI-Frameworks/blob/master/slides/IA_Framework_RS.pdf) / [TP](https://github.com/wikistat/AI-Frameworks/tree/master/RecomendationSystem)
   - ??
        * [Slides]() / [TP]()
- **Session 6 04-01-20**
   - Free time on project.

   
## Evaluation

The evaluation is associated to the [DEFI-IA](https://defi-ia.insa-toulouse.fr/)

### Objective
    
You will be evaluated on your capacity of acting like a ***Data Scientist***, i.e. 

* Handle a new dataset and explore it.
* Find a solution to address the defi's problem with a high score (above baseline).
* Explain the choosen algorithm.
* Write a complete pipeline to easily reproduce the results.
* Justify the choice of the algorithms and the environment (CPU/GPU, Cloud etc..).
* Share it and make your results easily reproducible (Git -  docker, conda environment.).

### Notations

1. Project - (**60%**): a Git repository.
    * The git should contain a clear markdown Readme, which describes  (**33%**)
        * Which result you achieved? In which computation time? On which engine?
        * What do I have to install to be able to reproduce the code? 
        * Which command do I have to run to reproduce the results?
    * The code has to be easily reproducible.  (**33%**)
        * Packages required has to be well described.
         (a **requirements.txt** files is the best)
        * Conda command or docker command can be furnish 
    * The code should be clear and easily readable. (**33%**)
        * Final results can be run in a script and not a notebook.
        * Only final code can be found in this script. 
    * **Deadline** :  January ?? 2020.
    
2. Oral presentation - (**40%**):
    * Quality of the presentation. **25%**
    * In-Deep explanation of the chosen algorithm. **25%**
    * Choice of the tools-infrastructure used. **25%**
    * Results you obtained. **25%**
    * **Date** : January ??, 2020. 

### Other details

 * Group of 4 to 5 people. (Deadline for group December ?? 2020)