# coding: utf-8

import project

## Create Project Manager
projectManager = project.ProjectManager(print_command = True, execute_command=False)
projectManager.set_image_container_names(image_name=TO COMPLETE, container_name=TO COMPLETE)

##;Start the instace
print("\n########## Start Instance ##########")
projectManager.instance_init()

## Update data
print("\n########## Create directories ##########")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_folder+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_results+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_model+ "'")
projectManager.execute_code_ssh("'mkdir " + projectManager.remote_code+ "'")

## Start container
print("\n########## Start container ##########")
projectManager.manage_container("run")


print("\n########## Send data and unzip it ##########")
projectManager.update_data("data.zip", container=True)

## Update last version of code
print("\n########## Send code ##########")
projectManager.update_code()

## Execute Job
print("\n########## Execute script ##########")
args = [["epochs", "2"], ["batch_size", "100"]]
projectManager.execute_python_script_container("learning_solution.py", args = args)
projectManager.execute_python_script_container("prediction_solution.py", args = args)

## Collect job output
print("\n########## Get Result back ##########")
projectManager.collect_results()

## stop and remove container
print("\n########## Stop & remove container ##########")
projectManager.manage_container("stop")
projectManager.manage_container("remove")

## Finalize instance
print("\n########## Stop Instance ##########")
projectManager.instance_end()








