
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
from scipy.fftpack import fft

# High pass filter

import numpy as np
from scipy.signal import butter, filtfilt

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


# High pass filter
# Acc = (Gravity + UserAcc)
# Annotation for dominent frequency

def plot_movement_grav_HF_anno(file_path, sample_rate=50):
    df = pd.read_csv('./Motion/'+file_path)
    df['label'] = file_path[:3]
    time = np.linspace(0, len(df)/sample_rate, len(df))

    fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # 3 rows for x, y, z axes and 4 columns for the four graphs
    
    # title coming from the file_path format (jog_16/sub_7.csv)
    activity = file_path.split('/')[0].upper()
    person = file_path.split('/')[-1].upper()[:5]
    full_title = f"{activity} ({person})"
    fig.suptitle(full_title,fontsize=40)
    
    for i, axis in enumerate(['x', 'y', 'z']):
        df[f'Acceleration.{axis}'] = df[f'gravity.{axis}']+df[f'userAcceleration.{axis}']
        # acceleration = df[f'Acceleration.{axis}']
        acceleration = highpass_filter(df[f'Acceleration.{axis}'], cutoff=0.1, fs=50)
        velocity = cumtrapz(acceleration, dx=1/sample_rate, initial=0)
        position = cumtrapz(velocity, dx=1/sample_rate, initial=0)
        sp = np.fft.fft(acceleration)
        freq = np.fft.fftfreq(len(sp), 1 / sample_rate)
        
        data = [acceleration, velocity, position, sp]  # List of data for the four graphs
        titles = ['Acceleration', 'Velocity', 'Position', 'Frequency Analysis']
        ylabels = ['Acceleration (m/s^2)', 'Velocity (m/s)', 'Position (m)', 'Amplitude']
        
        for j in range(4):
            ax = axs[i, j]
            if j < 3:  # For Acceleration, Velocity, and Position
                ax.plot(time, data[j], label=f'{titles[j]} ({axis.upper()}-axis)')
                ax.set_xlabel('Time (s)')
                
            else:  # For Frequency Analysis
                ax.plot(freq[freq > 0], np.abs(data[j][freq > 0]), label=f'FFT ({axis.upper()}-axis)')
                ax.set_xlabel('Frequency (Hz)')

                # Additional lines to annotate dominant frequency and amplitude
                pos_freq = freq[freq > 0]
                pos_sp = np.abs(sp[freq > 0])
                
                dominant_frequency = pos_freq[np.argmax(pos_sp)]
                amplitude_at_dominant_frequency = np.max(pos_sp)
                
                ax.annotate(f'Dominant Frequency: {dominant_frequency:.2f} Hz\nAmplitude: {amplitude_at_dominant_frequency:.2f}', 
                xy=(dominant_frequency, amplitude_at_dominant_frequency), 
                xytext=(dominant_frequency + 1, amplitude_at_dominant_frequency),
                arrowprops=dict(facecolor='red', arrowstyle='->'),
                fontsize=12,
                color='red')
                ax.plot(freq[freq > 0], np.abs(data[j][freq > 0]), label=f'FFT ({axis.upper()}-axis)')
                ax.set_xlabel('Frequency (Hz)')
            ax.set_title(f'{titles[j]} ({axis.upper()}-axis)')
            ax.set_ylabel(ylabels[j])
            ax.legend()
    
    #plt.tight_layout()
    plt.show()

# Run the function on the dataset
plot_movement_grav_HF_anno('jog_9/sub_1.csv')