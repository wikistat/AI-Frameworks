# coding: utf-8

import project
import itertools

## Create Project Manager
projectManager = project.ProjectManager()
## Set up project variables
projectManager.load_yaml('ACOMPLETER/conf.yml')

## Initialize instance
projectManager.set_instance_name('ACOMPLETER')
projectManager.set_ssh_key_file('ACOMPLETER')

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


## Start container
projectManager.manage_container("run", "image_name", "container_name")


## Execute Job


args = [["epochs", "10"], ["batch_size", "100"]]


projectManager.execute_python_script_container("learning_solution.py", "sample_2", "container_name", args = args)
projectManager.execute_python_script_container("prediction_solution.py", "sample_2", "container_name", args = args)




## stop and remove container
projectManager.manage_container("stop", "image_name", "container_name")
projectManager.manage_container("remove", "image_name", "container_name")


## Collect job output
projectManager.collect_results()

## Finalize instance
projectManager.instance_end()
