# About the data format

The data series are in csv format with one header-line at the top.
In the **logmap_linear**, **logmap_tent**, and **rw_tent** data-sets there are two columns, the first column is the value of the time series and the second column is the coresponding indicator variable, whether the time-instance is anomalous or not.
The data-series in the ***sim_ecg** have an additional time axis as the first column.

An example readin:

```python
import pandas as pd
```


```python
data = pd.read_csv("./logmap_tent/0.csv")

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
      <th>is_anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.898227</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.356520</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.894712</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.367389</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.906416</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
