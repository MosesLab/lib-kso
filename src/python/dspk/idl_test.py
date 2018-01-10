from idlpy import IDL

import os

IDL.run('$ pwd', stdout=1)
cwd = os.path.dirname(os.path.realpath(__file__))
dir = cwd + '/../../idl/dspk"'
cmd = '!path = !path + ":' + dir
IDL.run(cmd)
IDL.run('print, !path', stdout=1)

IDL.despik()



