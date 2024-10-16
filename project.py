import numpy as np
import time
import matplotlib.pyplot as plt
from collections import deque

def simulate_data_stream():
    """
    Simulates a continuous stream of data with seasonal patterns and random noise.
    
    Yields:
    float: A simulated data point representing real-time data.
    """
    while True:
        time_step = time.time()  # Use the current time for continuous variation
        data_point = np.sin(time_step) + np.random.normal(0, 0.1)  # Sine wave with noise
        yield data_point
        time.sleep(0.1)  # Simulate real-time by adding delay

def update_mean_std(prev_mean, prev_std, new_data, old_data, window_size):
    """
    Updates the mean and standard deviation for a rolling window of data.
    
    Args:
    prev_mean (float): Previous mean of the window.
    prev_std (float): Previous standard deviation of the window.
    new_data (float): New data point being added to the window.
    old_data (float): Old data point being removed from the window.
    window_size (int): Size of the rolling window.
    
    Returns:
    tuple: Updated mean and standard deviation.
    """
    # Update mean by removing old_data and adding new_data
    new_mean = prev_mean + (new_data - old_data) / window_size
    
    # Update variance, then calculate new standard deviation
    new_variance = (prev_std ** 2 * (window_size - 1) - (old_data - prev_mean) ** 2 + (new_data - new_mean) ** 2) / window_size
    
    # Ensure variance is non-negative
    if new_variance < 0:
        new_variance = 0
    
    new_std = np.sqrt(new_variance)
    
    return new_mean, new_std

def detect_and_visualize_anomalies(data_stream, window_size=30, threshold=3):
    """
    Detects anomalies in real-time using the Z-Score method and visualizes the data stream.
    
    Args:
    data_stream (generator): A generator yielding real-time data points.
    window_size (int): The size of the rolling window for calculating mean and standard deviation.
    threshold (float): Z-Score threshold for detecting anomalies.
    
    Returns:
    None
    """
    data_window = deque(maxlen=window_size)  # Rolling window to store recent data points
    mean, std_dev = 0, 0  # Initialize mean and standard deviation
    
    # Enable interactive mode for real-time plotting
    plt.ion()
    fig, ax = plt.subplots()
    
    stream_data = []  # To store the data stream for visualization
    anomaly_points = []  # To store detected anomalies for plotting
    
    for data_point in data_stream:
        if len(data_window) == window_size:
            old_data = data_window[0]
            mean, std_dev = update_mean_std(mean, std_dev, data_point, old_data, window_size)
            
            # Calculate Z-Score for the current data point
            z_score = (data_point - mean) / std_dev if std_dev != 0 else 0
            
            # Store data for visualization
            stream_data.append(data_point)
            if abs(z_score) > threshold:
                anomaly_points.append(data_point)  # Mark as anomaly
                print(f"Anomaly detected! Data point: {data_point:.4f}, Z-Score: {z_score:.4f}")
            else:
                anomaly_points.append(np.nan)  # Normal point (not an anomaly)
            
            # Update the plot with the current data and anomalies
            ax.clear()
            ax.plot(stream_data, label="Data Stream")
            ax.scatter(range(len(anomaly_points)), anomaly_points, color='red', label="Anomalies")
            ax.legend()
            plt.draw()
            plt.pause(0.01)  # Pause to ensure smooth real-time plotting
        
        # Add new data point to the window
        data_window.append(data_point)
        
        # Initialize mean and std when the window is full
        if len(data_window) == window_size and mean == 0 and std_dev == 0:
            mean, std_dev = np.mean(data_window), np.std(data_window)

# Run the anomaly detection without a stopping condition (continuous stream)
if __name__ == "__main__":
    stream = simulate_data_stream()  # Continuous data stream
    detect_and_visualize_anomalies(stream)
