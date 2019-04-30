import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

step_eeg = 10
step_emg = 10
max_range = 500
filename = 'latencyHeatmap_eeg%i_emg%i.npy'%(step_eeg, step_emg)

# heatMap is the to be saved numpy array of mean squared error 
# for different eeg and emg latencies
heatMap = np.load(filename)
print(heatMap)
idx_min = np.where(heatMap == heatMap.min())
min_eeg_idx, min_emg_idx = idx_min[0][0], idx_min[1][0]

eeg_latency = min_eeg_idx * step_eeg
emg_latency = min_emg_idx * step_emg
print("The eeg latency for the smallest mse in the heatmap: %i" %(eeg_latency))
print("The emg latency for the smallest mse in the heatmap: %i" %(emg_latency))
# row names:
eegDelays = [x for x in range(0, max_range, step_eeg)]
# col names
emgDelays = [x for x in range(0, max_range, step_emg)]



z = []

for row_idx, eegDelay in enumerate(eegDelays):
    new_row = []
    for col_idx, emgDelay in enumerate(emgDelays):
        new_row.append(heatMap[row_idx, col_idx])
    z.append(list(new_row))

data = [
    go.Heatmap(
        z=z,
        x=emgDelays,
        y=eegDelays,
        colorscale='Viridis',
    )
]

layout = go.Layout(
    title='Force-EEG and EEG-EMG Latency Heatmap',
    xaxis = dict(ticks='', nticks=36),
    yaxis = dict(ticks='' )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='latency-heatmap')
