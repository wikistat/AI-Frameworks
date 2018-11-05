# coding: utf-8

import project
import itertools

## Create Project Manager
projectManager = project.ProjectManager()

## Set up project variables
projectManager.load_yaml('/Users/bguillouet/Insa/TP_Insa/dev/IA-Frameworks/tp_google_cloud/conf.yml')

## Initialize instance
projectManager.set_instance_name('instance-2')
projectManager.instance_init()



## Update data
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_folder+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_data+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_results+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_model+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_code+ "'")

projectManager.update_data("sample_2.zip")


## Update last version of code
projectManager.update_code()


## Execute Job
args = [["epochs", "10"], ["batch_size", "100"]]


projectManager.execute_python_script("learning_solution.py", "sample_2", args = args)
projectManager.execute_python_script("prediction_solution.py", "sample_2", args = args)



## Collect job output
projectManager.collect_results()

## Finalize instance
#projectManager.instance_end()
