# coding: utf-8

import project

## Create Project Manager
projectManager = project.ProjectManager()

##;Start the instace
print("\n########## Start Instance ##########")
projectManager.instance_init()

## Update data
print("\n########## Create directories ##########")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_folder+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_results+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_model+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_code+ "'")

print("\n########## Send data and unzip it ##########")
projectManager.update_data("data.zip")

## Update last version of code
print("\n########## Send code ##########")
projectManager.update_code()

## Execute Job
print("\n########## Execute script ##########")
args = [["epochs", "10"], ["batch_size", "100"]]
projectManager.execute_python_script("learning.py", args = args)
projectManager.execute_python_script("prediction.py", args = args)

## Collect job output
print("\n########## Get Result back ##########")
projectManager.collect_results()

## Finalize instance
print("\n########## Stop Instance ##########")
projectManager.instance_end()
