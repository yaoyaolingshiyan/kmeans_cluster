import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt

c = colors.cnames.keys()
c_dark = list(filter(lambda x: x.startswith('dark'), c))
print(c_dark)

x = np.asarray([i for i in range(10)])
# x = np.linspace(-1,1,50)
print(type(x))
y = 2 * x

plt.plot(x, y, color=c_dark[0], label='my test')
plt.legend(loc='upper left')
plt.xlabel('cluster_num')
plt.ylabel('accuracy')
plt.show()
