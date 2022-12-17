import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys

infile = sys.argv[1]
df = pd.read_csv(infile)
print(df)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(2, 1, 1)
ax.plot(df.k, np.abs(df.p), '.', label='|p|')
ax.plot(df.k, np.abs(df.r), '.', label='|r|')
ax.plot(df.k, np.abs(df.x), '.', label='|x|')
ax.set_yscale('log')
ax.grid(True)
ax.legend()
ax.set_xlim([0, len(df)])
ax.set_ylabel('vector norm')

ax = fig.add_subplot(2, 1, 2)
ax.plot(df.k, np.abs(df.a), '.', label=r'$ \alpha $')
ax.plot(df.k, np.abs(df.b), '.', label=r'$\beta$')
ax.set_yscale('log')
ax.grid(True)
ax.legend()
ax.set_xlim([0, len(df)])
ax.set_ylabel('coefficient')

ax.set_xlabel('iteration step')
plt.savefig('cg.png')
