# coding: utf-8

import os
import subprocess


def command_application(command, command_only):
    """

    :param command:
    :param command_only:
    :return:
    """

    if command_only:
        print(command)
    else:
        os.system(command)


class InstancesManager:

    zone = 'europe-west1-b'
    command_only = False
    instance_template = 'ref-dev-dl-k80-template'

    instance_name = 'undefined'
    status = 'UNDEFINED'

    ssh_key_file = 'undefined'

    # Function that lists instances in given project
    def list(self):
        """

        :return:
        """
        command = 'gcloud compute instances list'

        command_application(command, self.command_only)

    # Set instance name
    def set_instance(self, instance_name):
        """

        :param instance_name:
        :return:
        """

        self.instance_name = instance_name

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

    # Function that creates new GPU instance from  instance template
    def create(self, instance_name):
        """

        :param instance_name:
        :return:
        """

        self.set_instance(instance_name)

        command = 'gcloud compute instances create ' + \
                  self.instance_name #TODO: complete command with instance template

        command_application(command, self.command_only)

    def delete(self):
        """

        :return:
        """

        command = 'gcloud compute instances delete ' + \
                  self.instance_name + ' --zone ' + self.zone

        command_application(command, self.command_only)

    def start(self):
        """

        :return:
        """

        command = 'gcloud compute instances start ' + self.instance_name + ' --zone ' + self.zone

        command_application(command, self.command_only)

    def stop(self):
        """

        :return:
        """

        command = 'gcloud compute instances stop ' + self.instance_name + ' --zone ' + self.zone

        command_application(command, self.command_only)

    def ssh_command(self, ssh_command):
        """

        :param ssh_command:
        :return:
        """

        command = 'gcloud compute ssh ' + self.instance_name \
                  + ' --zone ' + self.zone \
                  + ' --command ' + ssh_command \
                  + ' --ssh-key-file ' + self.ssh_key_file

        print(command)

        command_application(command, self.command_only)

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

        print(command)

        command_application(command, self.command_only)

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

        print(command)

        command_application(command, self.command_only)
