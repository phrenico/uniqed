from tofpy.models.tof import TOF
import numpy as np
import matplotlib.pyplot as plt


X = np.random.rand(10000).reshape([1000, 10])

mytof = TOF(cutoff_n=10)
y = mytof.fit(X).predict(X)
tof_score = 1/mytof.outlier_score_

plt.figure()
plt.plot(tof_score)
plt.show()