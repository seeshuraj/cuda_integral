import matplotlib.pyplot as plt

# Benchmark data from final runs
sizes = ['5000×5000', '8192×8192', '16384×16384', '20000×20000']
cpu_times = [0.5289, 1.3160, 5.2538, 8.4552]
gpu_times = [0.5259, 0.3723, 1.4670, 2.3334]
speedups = [cpu/gpu for cpu, gpu in zip(cpu_times, gpu_times)]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(sizes, speedups, marker='o', linestyle='-', linewidth=2)
plt.title('GPU Speedup over CPU for Exponential Integral')
plt.xlabel('Problem Size (n × m)')
plt.ylabel('Speedup (CPU Time / GPU Time)')
plt.grid(True)
plt.tight_layout()
plt.savefig('speedup_plot.png')
plt.show()
