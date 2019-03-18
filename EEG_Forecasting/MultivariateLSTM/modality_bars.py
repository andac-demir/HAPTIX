import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np

eeg = np.array([0.00000999, 0.00000977, 0.00000588, 0.00000658, 0.00000576,
                0.00000624, 0.00000635, 0.00000632, 0.00000745, 0.00000677,
                0.00000558, 0.00000622, 0.00000509, 0.00000669])
eeg_emg = np.array([0.00001199, 0.00001178, 0.00000608, 0.00000742, 0.00000644,
                    0.00000627, 0.00000644, 0.00000653, 0.00001049, 0.00000664,
                    0.00000653, 0.00001049, 0.00000664, 0.00000605, 0.00000600,
                    0.00000555, 0.00000730])
eeg_force = np.array([0.00001049, 0.00001220, 0.00000600, 0.00000642, 
                      0.00000595, 0.00000611, 0.00000657, 0.00000715, 
                      0.00000800, 0.00000667, 0.00000558, 0.00000602, 
                      0.00000524, 0.00000736])
eeg_emg_force = np.array([0.00001115, 0.00001191, 0.00000615, 0.00000670, 
                          0.00000651, 0.00000773, 0.00000608, 0.00000682, 
                          0.00001054, 0.00000783, 0.00000563, 0.00000627, 
                          0.00000534, 0.00000684])
force = np.array([0.00002056,0.00002228,0.00001256,0.00001593,0.00001448,
                  0.00001797,0.00001522,0.00001038,0.00002141,0.00001574,
                  0.00001234,0.00001179,0.00001393,0.00001923])
emg = np.array([0.00002381,0.00002513,0.00001185,0.00001567,0.00001550,
                0.00001559,0.00001666,0.00000982,0.00002104,0.00001465,
                0.00001311,0.00001064,0.00001152,0.00001470])
force_emg = np.array([0.00002308,0.00002005,0.00001133,0.00001606,0.00001474,
                      0.00001629,0.00001593,0.00000947,0.00001971,0.00001547,
                      0.00001320,0.00001044,0.00001281,0.00001898])

trace0 = go.Box(
    y=eeg,
    name = 'EEG only',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=emg,
    name = 'EMG only',
    marker = dict(
        color = 'rgb(255, 133, 27)',
    )
)
trace2 = go.Box(
    y=force,
    name = 'Force only',
    marker = dict(
        color = 'rgb(220, 20, 60)',
    )
)
trace3 = go.Box(
    y=eeg_emg,
    name = 'EEG and EMG',
    marker = dict(
        color = 'rgb(139, 0, 0)',
    )
)
trace4 = go.Box(
    y=eeg_force,
    name = 'EEG and Force',
    marker = dict(
        color = 'rgb(107, 174, 214)',
    )
)
trace5 = go.Box(
    y=force_emg,
    name = 'Force and EMG',
    marker = dict(
        color = 'rgb(169, 169, 169)',
    )
)
trace6 = go.Box(
    y=eeg_emg_force,
    name = 'EEG, EMG and Force',
    marker = dict(
        color = 'rgb(0, 0, 128)',
    )
)

data = [trace0, trace1, trace2, trace3, trace4, trace5, trace6]
py.iplot(data)
layout = go.Layout(title = "Modality Interpretation for Each Parameter"
                           " Based On RMSE")

fig = go.Figure(data=data,layout=layout)
py.iplot(fig, filename = "Modality Interpretation for Each Parameter"
                         " Based On RMSE")