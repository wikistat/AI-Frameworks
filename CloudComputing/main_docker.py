# coding: utf-8

import project

## Create Project Manager
projectManager = project.ProjectManager(print_command = True, execute_command=True)

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
projectManager.manage_container("run", "my_image", "my_container")


print("\n########## Send data and unzip it ##########")
projectManager.update_data("data.zip", container = "my_container")

## Update last version of code
print("\n########## Send code ##########")
projectManager.update_code()

## Execute Job
print("\n########## Execute script ##########")
args = [["epochs", "2"], ["batch_size", "100"]]
projectManager.execute_python_script_container("learning", "my_container", args = args)
projectManager.execute_python_script_container("prediction","my_container", args = args)

## Collect job output
print("\n########## Get Result back ##########")
projectManager.collect_results()

## stop and remove container
print("\n########## Stop & remove container ##########")
projectManager.manage_container("stop", "my_image", "my_container")
projectManager.manage_container("remove", "my_image", "my_container")

## Finalize instance
print("\n########## Stop Instance ##########")
projectManager.instance_end()


