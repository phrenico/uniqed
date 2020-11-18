from tofpy.data.gen_logmap import generate_logmapdata
from tofpy.runners.tof_run import detect_outlier
import matplotlib.pyplot as plt

# Generate some data
data_df = generate_logmapdata(rseed=359)

# Detect outliers
res_df = detect_outlier(data_df[['value']], cutoff_n=80)
print(res_df.head())


# plot the results
plt.figure()
plt.subplot(211)
plt.plot(res_df['value'], color='tab:blue', label='time series')
plt.plot(res_df['value'].loc[data_df.query("is_anomaly==1").index.values],
         color='tab:green', label='anomaly')
plt.plot(res_df.query("TOF==1")['value'], lw=0, marker='o',
         color='tab:orange', label='TOF')
plt.xlabel('t')
plt.ylabel('values')
plt.legend(loc='upper left', framealpha=1)

plt.subplot(212)
plt.plot(res_df['TOF_score'], color='tab:blue', label='time series')
plt.plot(res_df['TOF_score'].loc[data_df.query("is_anomaly==1").index.values],
         color='tab:green', label='anomaly')
plt.plot(res_df.query("TOF==1")['TOF_score'], lw=0, marker='o',
         color='tab:orange', label='TOF')
plt.show()