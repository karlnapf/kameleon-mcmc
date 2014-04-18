"""
This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or
(at your option) any later version.

Written (W) 2013 Heiko Strathmann
Written (W) 2013 Dino Sejdinovic
"""

import subprocess
class GitTools(object):
    @staticmethod
    def get_hash():
        return subprocess.check_output('git rev-parse HEAD'.split()).strip()

    @staticmethod
    def get_branch():
        return subprocess.check_output("git rev-parse --abbrev-ref HEAD".split()).strip()