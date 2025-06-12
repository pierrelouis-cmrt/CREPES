import matplotlib.pyplot as plt
import numpy as np

def generate_temperature_scale(min_temp=0, max_temp=388):
    temperatures = np.linspace(min_temp, max_temp, 100)
    fig, ax = plt.subplots(figsize=(6, 1))
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    ax.imshow(gradient, aspect='auto', cmap='viridis', extent=[min_temp, max_temp, 0, 1])
    ax.set_xticks(np.arange(min_temp, max_temp + 1, 20))
    ax.set_yticks([])

    ax.set_title('Échelle des températures (°K)')
    plt.show()

generate_temperature_scale()
