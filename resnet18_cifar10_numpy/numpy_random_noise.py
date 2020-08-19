import numpy as np
import pdb


faul_map = np.empty(11169152, dtype="uint32")

for i in range(faul_map.size):
    prob_bi = np.random.choice([0, 1], size=32, p=(0.7, 0.3))
    prob_8 = np.packbits(prob_bi)
    prob32 = (
        (prob_8[0].astype("uint32") << 24)
        +(prob_8[1].astype("uint32") << 16)
        +(prob_8[2].astype("uint32") << 8)
        + (prob_8[3].astype("uint32"))
    )
    faul_map[i] = prob32

