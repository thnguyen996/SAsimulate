import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np 
import matplotlib.ticker as mticker
matplotlib.use('tkagg')
with open('data.json') as file:
    file_data = json.load(file)


data = []
for i in range(len(file_data)):
    data.append(file_data[i][2])

x_ax = np.linspace(0.0000000001, 6.9*0.00001, 70)
print(data, x_ax)
plt.rc('font', family='serif')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(x_ax, data)

f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(g))

ax.set_xlabel('Error rate')
ax.set_ylabel('Classification error')
ax.set_xlim(left=1e-10)
plt.show()
