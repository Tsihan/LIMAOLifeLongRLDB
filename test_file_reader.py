import glob
import os
print(os.getcwd())
path_glob = './data/replay-Balsa_JOBSlowSplit-*'
paths = glob.glob(os.path.expanduser(path_glob))
print(paths)

test_paths = glob.glob('./data/IMDB_assorted_small/replay-Balsa_JOBRandSplit*')
print(test_paths)