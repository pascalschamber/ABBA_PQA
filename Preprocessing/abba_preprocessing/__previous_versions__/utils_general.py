import os
from pathlib import Path

def verify_outputdir(adir):
    # make an outputdir if it doesn't exists
    if not os.path.exists(adir):
        os.mkdir(adir)
    return adir