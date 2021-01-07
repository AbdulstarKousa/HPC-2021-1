

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
#%%Plotting for Assigment 1, 02614


#cache sizes 
# L1i 32K
# L1d 32K = 32K -> line 
# L2 256K = 330K -> line 
# L3 30720K = 

#Getting data
filename='matmult_2.dat'

Data = np.loadtxt('matmult_2.dat', comments= "%", dtype=str)

mem = Data[:,0].astype(np.float)
Mflops = Data[:,1].astype(np.float)
per = Data[:,4].astype(np.str) #OBS new file change to 3 

permutations = np.unique(per)

marker = itertools.cycle(('8', '+', '.', 'o', '<','s','h')) 

for x in permutations:
    index = per == x       
    plt.plot(mem[index], Mflops[index],label=x,linestyle='--', marker=next(marker))

plt.legend()
plt.xlabel('Memory in kB')
plt.ylabel('MFlops/s')
plt.title('Hejsa')
plt.grid()
plt.xscale("log")
plt.axvline(x=32)
#plt.axvline(x=64)
plt.axvline(x=288)
#plt.axvline(x=330)
plt.axvline(x=30720)
plt.show()

