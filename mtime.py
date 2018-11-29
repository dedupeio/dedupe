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

import subprocess, shlex
import sys, os.path

if os.utime in getattr(os, 'supports_follow_symlinks', []):
    def lutime(path, times):
        os.utime(path, times, follow_symlinks=False)
else:
    def lutime(path, times):
        if not os.path.islink(path):
            os.utime(path, times)

# List files matching user pathspec, relative to current directory
filelist = set()
for path in (sys.argv[1:] or [os.path.curdir]):

    # file or symlink (to file, to dir or broken - git handles the same way)
    if os.path.isfile(path) or os.path.islink(path):
        filelist.add(os.path.relpath(path))

    # dir
    elif os.path.isdir(path):
        for root, subdirs, files in os.walk(path):
            if '.git' in subdirs:
                subdirs.remove('.git')

            for file in files:
                filelist.add(os.path.relpath(os.path.join(root, file)))

# Process the log until all files are 'touched'
mtime = 0
gitobj = subprocess.Popen(shlex.split('git whatchanged --pretty=%at'),
                          stdout=subprocess.PIPE)
for line in gitobj.stdout:
    line = line.strip()

    # Blank line between Date and list of files
    if not line: continue

    # File line
    if line.startswith(':'):
        file = os.path.normpath(line.split('\t')[-1])
        if file in filelist:
            filelist.remove(file)
            #print mtime, file
            lutime(file, (mtime, mtime))

    # Date line
    else:
        mtime = long(line)

    # All files done?
    if not filelist:
        break
