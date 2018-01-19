from idlpy import IDL

import os

def dspk_idl(data,std_dev=4.5, Niter=10):

    # IDL.run('$ pwd', stdout=1)
    cwd = os.path.dirname(os.path.realpath(__file__))
    dir = cwd + '/../../idl/dspk"'
    cmd = '!path = !path + ":' + dir
    IDL.run(cmd)
    # IDL.run('print, !path', stdout=1)

    return IDL.despik(data, sigmas=std_dev, Niter=Niter)



