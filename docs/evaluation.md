# Evaluation

The evaluation is associated to the [DEFI-IA](http://www.kaggle.com/competitions/defi-ia-2023)  
(Introduction video [link](https://cloud.irit.fr/index.php/s/PFOyTUqe2TrbIAf))
### Objectives
    
You will be evaluated on your capacity of acting like a ***Data Scientist***, i.e. 

* Collect the data.
* Doing some exploratory analysis.
* Create new features.
* Write a complete pipeline to train and test your models.
* Justify your modelisation choices.
* Interpret your results.
* Work in group (Git).
* Share it and make your results easily reproducible (Docker, Gradio).

### Evaluation criteria

You are expected to produce a code that is easily readable and reproducible.  
Your code should at leat contain the three following files (but you are ecouraged to add more to make it more readable):  
    * `train.py` : the training script  
    * `app.py` : code to launch a gradio application to test your model (see [Gradio](https://gradio.app/))  
    * `analysis.ipynb` : a notebook containing your exploratory analysis and interpretability results on your model.  
    * `Dockerfile` : a Dockerfile to build a docker image of your application (see [Docker](https://www.docker.com/))

You will be evaluated on the following criteria:

1. Project - (**70%**):  
You must provide a git repository with a complete history of your commits.  
Your capacity to work in group will be evaluated, your commit history must contain commits from several users at different dates.  
You must provide a Dockerfile to build a docker image that can be used to run your code (training and the Gradio application).  
The git should contain a clear markdown Readme, which describes:  
    *   the result you achieved  
    *   the commands to run for training your model or launching the gradio application (from a docker container)  
        
The code should be clear and easily readable. 
No notebooks exept for the exploratory analysis.
        * 
    <!-- * **Deadline** :  January 29 2021. -->
    
2. Oral presentation - (**30%**)  
15 minutes presentation + 10 minutes questions.  
You will be evaluated on the following criteria:
    * Quality of the presentation. 
    * Explanations of the chosen features and algorithm. 
    * Demonstration of your application. 
    * Some insights on your model biais and interpretability. 
    <!-- * **Date** : January 29, 2021.  -->

### Other details

 * Group of 4 people (DEFI IA's team).

 