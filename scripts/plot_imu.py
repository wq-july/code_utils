import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime, timedelta

def read_imu_data(file_path, start_ts=None, end_ts=None, invert_axes=None):
    data = np.genfromtxt(file_path, skip_header=1)
    # Convert nanoseconds to datetime objects
    timestamps = np.array([datetime.fromtimestamp(ts / 1e9) for ts in data[:, 0]])

    if start_ts is not None and end_ts is not None:
        start_datetime = datetime.fromtimestamp(start_ts)
        end_datetime = datetime.fromtimestamp(end_ts)
        # Filter data within the given timestamps range
        filtered_indices = (timestamps >= start_datetime) & (timestamps <= end_datetime)
        data = data[filtered_indices]
        # Adjust timestamps to start from the first record of the filtered data
        timestamps = timestamps[filtered_indices]

    # Convert timestamps to seconds since the first record
    data[:, 0] = np.array([(ts - timestamps[0]).total_seconds() for ts in timestamps])
    
    if invert_axes:
        for axis in invert_axes:
            data[:, axis] *= -1

    return data

def plot_imu_data(data1, data2, label1, label2):
    labels = [os.path.splitext(label1)[0], os.path.splitext(label2)[0]]
    components = ['X', 'Y', 'Z']
    titles = ['Acceleration', 'Gyroscope']
    data_indices = [[4, 5, 6], [1, 2, 3]]  # Indexes for acc and gyr data in file

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

    for i, title in enumerate(titles):
        for j, comp in enumerate(components):
            index = data_indices[i][j]
            axs[i, j].plot(data1[:, 0], data1[:, index], label=f'{labels[0]} {comp}')
            axs[i, j].plot(data2[:, 0], data2[:, index], label=f'{labels[1]} {comp}')
            axs[i, j].set_title(f'{title} in {comp}')
            axs[i, j].set_xlabel('Time (seconds)')
            axs[i, j].set_ylabel(f'{title} ({"m/s^2" if title == "Acceleration" else "rad/s"})')
            axs[i, j].legend()

    plt.tight_layout()
    plt.show()

def plot_gyroscope_z(data1, data2, label1, label2):
    # Gyroscope Z index is 3
    plt.figure(figsize=(10, 5))
    plt.plot(data1[:, 0], data1[:, 3], label=f'{os.path.splitext(label1)[0]} Z')
    plt.plot(data2[:, 0], data2[:, 3], label=f'{os.path.splitext(label2)[0]} Z')
    plt.title('Gyroscope Z-axis Angular Velocity')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py path_to_imu_info1.txt path_to_imu_info2.txt [start_timestamp] [end_timestamp]")
        sys.exit(1)
    
    file_path1 = sys.argv[1]
    file_path2 = sys.argv[2]
    start_ts = float(sys.argv[3]) if len(sys.argv) > 3 else None
    end_ts = float(sys.argv[4]) if len(sys.argv) > 4 else None
    label1 = os.path.basename(file_path1)
    label2 = os.path.basename(file_path2)

    data1 = read_imu_data(file_path1, start_ts, end_ts)
    data2 = read_imu_data(file_path2, start_ts, end_ts, invert_axes=[4, 2, 3])
    plot_imu_data(data1, data2, label1, label2)
    plot_gyroscope_z(data1, data2, label1, label2)
