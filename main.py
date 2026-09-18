import sys, os
if sys.stdout is None: sys.stdout=open(os.devnull,'w')
if sys.stderr is None: sys.stderr=open(os.devnull,'w')
from gui.main_window import main
if __name__=='__main__': main()
