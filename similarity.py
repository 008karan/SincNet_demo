import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
di1=np.load("d_vect_speaker_f.npy",allow_pickle=True).item()

#a=[di1["3"]]
a=list(di1.values())
di2=np.load("d_vect_speaker_m.npy",allow_pickle=True).item()
#b=[di2["test5"]]
b=list(di2.values())
print(cosine_similarity(a,b))
