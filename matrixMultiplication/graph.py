import matplotlib.pyplot as plt
import numpy as np
import os

# Create a folder named 'graphs' to save the plot
if not os.path.exists('graphs'):
    os.makedirs('graphs')

size = [100,200,300,400,500]
cpu = [4.585 ,46.255,117.534,304.02,462.958]
gpu = [0.039 ,0.032 ,0.045,0.046,0.041]
# parallel = [x / 1_000_000 for x in parallel]
# serial = [x / 1_000_000 for x in serial]
#Set the figure size and font size
plt.figure(figsize=(6, 4))
plt.rcParams.update({'font.size': 10})

# Plot the data with a solid line and marker
plt.plot(size, cpu, '--s', label="CPU Matrix Multiplication", color='red', markersize=8, linewidth=2)
plt.plot(size, gpu, ':^', label="GPU Matrix Multiplication", color='blue', markersize=8, linewidth=2)

# Add axis labels and title
plt.xlabel('Dimensions of Random Matrix')
plt.ylabel('Execution Time (ms)')
plt.suptitle("GPU Vs. CPU Square Matrix Multiplication")
plt.title("Threads per Block = 512")

# Add gridlines and legend
plt.grid(False)
plt.legend(loc='best')


# # Save the plot to the 'graphs' folder with the name 'plot.png'
plt.savefig('graphs/CPUGPUMATRIX.png',dpi=600)