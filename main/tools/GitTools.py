import subprocess
class GitTools(object):
    @staticmethod
    def get_hash():
        return subprocess.check_output('git rev-parse HEAD'.split()).strip()

    @staticmethod
    def get_branch():
        return subprocess.check_output("git rev-parse --abbrev-ref HEAD".split()).strip()