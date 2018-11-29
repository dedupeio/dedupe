#!/usr/bin/env python
# Change mtime of files based on commit date of last change
#
#    Copyright (C) 2012 Rodrigo Silva (MestreLion) <linux@rodrigosilva.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Bare-bones version. Current dir must be top-level of work tree.
# Usage: git-restore-mtime-bare [pathspecs...]
#
# By default update all files
# Example: to only update only the README and files in ./doc:
# git-restore-mtime-bare README doc


import subprocess
import shlex
import sys
import os.path

filelist = set()
for path in (sys.argv[1:] or [os.path.curdir]):
    if os.path.isfile(path) or os.path.islink(path):
        filelist.add(os.path.relpath(path))
    elif os.path.isdir(path):
        for root, subdirs, files in os.walk(path):
            if '.git' in subdirs:
                subdirs.remove('.git')
            for file in files:
                filelist.add(os.path.relpath(os.path.join(root, file)))

mtime = 0
gitobj = subprocess.Popen(shlex.split('git whatchanged --pretty=%at'),
                          stdout=subprocess.PIPE)
for line in gitobj.stdout:
    line = line.decode()
    line = line.strip()
    if not line:
        continue

    if line.startswith(':'):
        file = line.split('\t')[-1]
        if file in filelist:
            filelist.remove(file)
            os.utime(file, (mtime, mtime))
    else:
        mtime = int(line)

    # All files done?
    if not filelist:
        break
