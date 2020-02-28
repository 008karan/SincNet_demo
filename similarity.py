import numpy as np
import json
import sys
from sklearn.metrics.pairwise import cosine_similarity
d1=sys.argv[1]
d2=folder=sys.argv[2]
#di1=np.load("d_vect_speaker_f.npy",allow_pickle=True).item()
di1=np.load(d1,allow_pickle=True).item()
#a=[di1["3"]]
a=list(di1.values())
#di2=np.load("d_vect_speaker_m.npy",allow_pickle=True).item()
di2=np.load(d2,allow_pickle=True).item()
#b=[di2["test5"]]
b=list(di2.values())
print(cosine_similarity(a,b))
