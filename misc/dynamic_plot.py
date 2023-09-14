import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create a figure and axis
fig, ax = plt.subplots()
x_data = np.linspace(0, 2 * np.pi, 100)  # X-axis data (0 to 2*pi)
line, = ax.plot(x_data, np.sin(x_data))  # Initial plot of sine wave

def update(frame):
    # Update the data for the sine wave
    line.set_ydata(np.sin(x_data + frame * 0.1))  # Update with a phase shift
    return line,

# Create an animation
ani = FuncAnimation(fig, update, frames=range(100), interval=100)  # 100 frames, 100ms delay between frames

plt.show()
