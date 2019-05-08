# coding: utf-8

import instances
import os
from yaml import load


class ProjectManager:

    # Local variables
    local_data = ''
    local_code = ''
    local_results = ''

    # Remote variables
    remote_data = ''
    remote_code = ''
    remote_results = ''

    # Main script
    main_script = ''

    # Instance
    instance = instances.InstancesManager()

    def load_yaml(self, GCP_conf_file):
        """

        :param GCP_conf_file:
        :return:
        """
        stream = open(GCP_conf_file, 'r')
        data = load(stream)

        self.local_data = data['local_data']
        self.local_code = data['local_code']
        self.local_results = data['local_results']

        self.remote_folder = data['remote_folder']
        self.remote_data = data['remote_data']
        self.remote_code = data['remote_code']
        self.remote_model = data['remote_model']
        self.remote_results = data['remote_results']

        self.container_folder = data['container_folder']
        self.container_data = data['container_data']
        self.container_code = data['container_code']
        self.container_model = data['container_model']
        self.container_results = data['container_results']

        self.main_script = data['main_script']

    def set_instance_name(self, instance_name):
        """

        :param instance_name:
        :return:
        """

        self.instance.instance_name = instance_name

    def set_ssh_key_file(self, ssh_key_file):
        """

        :param instance_name:
        :return:
        """

        self.instance.ssh_key_file = ssh_key_file

    def instance_init(self):
        """

        :return:
        """

        self.instance.instance_init()


    def update_data(self, zip_file):
        """

        :return:
        """

        self.instance.scp(direction='up',src_folder=self.local_data+"/"+zip_file, dst_folder=self.remote_data,
                          recurse=False, python_filter=False)
        # projectManager.update_data()
        self.instance.ssh_command("'unzip -o "+self.remote_data+"/"+zip_file+" -d "+self.remote_data+" | pv -l >/dev/null'")


    def refresh_data(self):
        """

        :return:
        """

        #TODO:data refresh
        return 0

    def update_code(self):
        """

        :return:
        """

        self.instance.scp(direction='up',
                          src_folder=self.local_code,
                          dst_folder=self.remote_code,
                          recurse=False,
                          python_filter=True)

    def execute_code_ssh(self, command):
        """

        :return:
        """

        ssh_command = command

        self.instance.ssh_command(ssh_command)

    def execute_code_ssh_container(self, command, container_id):
        """

        :return:
        """

        ssh_command = command

        self.instance.ssh_command_container(ssh_command, container_id)



    def execute_python_script(self, script_name, sample_dir, args=None):
        """

        :return:
        """

        command = "'python3 " + self.remote_code + "/" + script_name + " --data_dir " \
                  + self.remote_data+"/"+sample_dir + " --results_dir " + self.remote_results + " --model_dir " + self.remote_model

        if not(args is None):
            for k,v in args:
                command +=  " --" + k + " " + v
        command += "'"
        ssh_command = command


        self.instance.ssh_command(ssh_command)


    def execute_python_script_container(self, script_name, sample_dir, container_id, args=None):
        """

        :return:
        """

        command = "'python3 " + self.container_code + "/" + script_name + " --data_dir " \
                  + self.container_data+"/"+sample_dir + " --results_dir " + self.container_results + " --model_dir " + self.container_model

        if not(args is None):
            for k,v in args:
                command +=  " --" + k + " " + v
        command += "'"
        ssh_command = command


        self.instance.ssh_command_container(ssh_command, container_id)

    def manage_container(self, action, image_name, container_name, docker_dir ="/root/vm_dir/"):
        """

        :return:
        """
        if action =="run":
            command = "'sudo nvidia-docker run -t -d --name "+container_name+" -v " + self.remote_folder +":"+docker_dir +" " + image_name +"'"
        elif action == "stop":
            command = "'sudo nvidia-docker stop "+ container_name +"'"
        elif action == "remove":
            command = "'sudo nvidia-docker rm "+ container_name +"'"

        else:
            raise ValueError("'action' parameter should be 'run', 'stop' or 'remove' ")

        ssh_command = command

        self.instance.ssh_command(ssh_command)

    def collect_results(self):
        """

        :return:
        """

        self.instance.scp(direction='down',
                          src_folder=self.remote_results,
                          dst_folder=self.local_results,
                          recurse=True,
                          python_filter=False,
                          pickle_filter=True,
                          csv_filter=False)

        self.instance.scp(direction='down',
                          src_folder=self.remote_results,
                          dst_folder=self.local_results,
                          recurse=True,
                          python_filter=False,
                          pickle_filter=False,
                          csv_filter=True)

    def instance_end(self):
        """

        :return:
        """

        self.instance.instance_final()
