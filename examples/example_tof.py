from uniqed.models.tof import TOF
from uniqed.data.gen_logmap import generate_logmapdata
from uniqed.transformers.transformers import TimeDelayEmbedder
import matplotlib.pyplot as plt

# Generate some data
data = generate_logmapdata(rseed=231)
x = data['value'].values
t = data.index.values

# Time delay embedding of the time series
X = TimeDelayEmbedder().fit_transform(x)
T = TimeDelayEmbedder().fit_transform(t)

# Initialize TOF instance and find the anomaly
mytof = TOF(cutoff_n=100)
y = mytof.fit_predict(X)
tof_score = 1. / mytof.outlier_score_

# Plot the results
plt.figure()
plt.subplot(211)
plt.plot(data)
plt.legend(['time series', 'anomaly'], loc='upper left')
plt.ylabel('values')
plt.xlim(0, 2000)

plt.subplot(212)
plt.scatter(T[:, 0], tof_score, c=y)
plt.ylabel("TOF score")
plt.xlabel("t")
plt.xlim(0, 2000)
plt.grid(True)

plt.show()