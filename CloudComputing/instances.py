# coding: utf-8

import os
import subprocess





class InstancesManager:

    def __init__(self, print_command = True, execute_command=False):

        # VM arguments
        self.zone = 'us-east1-c' # TODO
        self.instance_name = 'instance-1'  # TODO
        self.ssh_key_file = '/Users/brendanguillouet/.ssh/google_compute_engine'  # TODO

        # Action arguments
        self.print_command = print_command
        self.execute_command = execute_command

        self.status = 'UNDEFINED'


    def command_application(self, command):
        """

        :param command:
        :param command_only:
        :return:
        """

        if self.print_command:
            print(command)
        if self.execute_command:
            os.system(command)


    # Check instance state : TERMINATED, STOPPING, STAGING, RUNNING
    def check_state(self):
        """

        :return:
        """

        command = 'gcloud compute instances describe '+self.instance_name+' --zone '+self.zone \
                  + ' | grep "status: RUNNING"'

        status = subprocess.getoutput(command)

        if status == 'status: RUNNING':
            self.status = "RUNNING"
        else:
            self.status = "TERMINATED"

    # Starts instance if not the case
    def instance_init(self):
        """

        :return:
        """

        self.check_state()

        if self.status is "TERMINATED":
            self.start()

    # Stops instance if not the case
    def instance_final(self):
        """

        :return:
        """

        self.check_state()

        if self.status is "RUNNING":
            self.stop()

    def start(self):
        """

        :return:
        """

        command = 'gcloud compute instances start ' + self.instance_name + ' --zone ' + self.zone

        self.command_application(command)

    def stop(self):
        """

        :return:
        """

        command = 'gcloud compute instances stop ' + self.instance_name + ' --zone ' + self.zone

        self.command_application(command)

    def ssh_command(self, ssh_command):
        """

        :param ssh_command:
        :return:
        """

        command = 'gcloud compute ssh ' + self.instance_name \
                  + ' --zone ' + self.zone \
                  + ' --command ' + ssh_command \
                  + ' --ssh-key-file ' + self.ssh_key_file

        self.command_application(command)

    def ssh_command_container(self, ssh_command, container_id):
        """

        :param ssh_command:
        :return:
        """

        command = 'gcloud compute ssh ' + self.instance_name \
                  + ' --zone ' + self.zone \
                  + ' --command ' + ssh_command \
                  + ' --ssh-key-file ' + self.ssh_key_file \
                  + ' --container ' + container_id

        self.command_application(command)

    def scp(self, direction, src_folder, dst_folder, recurse=True, python_filter=True, pickle_filter=False, csv_filter=False):
        """

        :param direction:
        :param src_folder:
        :param dst_folder:
        :param recurse:
        :param python_filter:
        :return:
        """

        command = 'gcloud compute scp '

        if recurse:
            command = command + '--recurse '

        command = command + '--zone '\
                  + self.zone+' '\

        if direction is 'up':
            dst_folder = self.instance_name+':'+dst_folder
        else:
            src_folder = self.instance_name+':'+src_folder

        # Adding source folder
        command = command + src_folder

        if python_filter:
            command = command + '/*.py'

        if pickle_filter:
            command = command + '/*.pkl'

        if csv_filter:
            command = command + '/*.csv'

        # Adding destination folder
        command = command + ' ' + dst_folder

        command = command + ' --ssh-key-file ' + self.ssh_key_file

        self.command_application(command)
