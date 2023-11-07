import tensorflow as tf
from tensorflow import keras
from xarray import DataArray
import matplotlib.pyplot as plt
from jetstream_hugo.plots import Clusterplot
from jetstream_hugo.definitions import get_region



class SOMCallback(keras.callbacks.Callback):
    def __init__(self, epochs, speed=1, start_sigma=2):
        super().__init__()
        self.epoch_number = epochs
        self.speed = speed
        self.start_sigma = tf.constant(start_sigma, dtype=tf.float32)

    def on_train_begin(self, logs=None):
        if self.start_sigma is None:
            map_size = self.model.som_layer.map_size
            self.start_sigma = max(map_size[0], map_size[1]) / 2
        self.tau = self.epoch_number / self.speed
        self.model.sigma = tf.Variable(self.start_sigma, trainable=False, name='sigma', dtype=tf.float32)

    def on_epoch_begin(self, epoch, logs=None):
        tf.keras.backend.set_value(self.model.sigma, self.start_sigma * tf.math.exp(-epoch / self.tau))
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['sigma'] = tf.keras.backend.get_value(self.model.sigma)
        
        
class PlotSOMCallback(keras.callbacks.Callback):
    def __init__(self, da: DataArray, path: str, **kwargs):
        super().__init__()
        if "time" in da.dims:
            da = da[0]
        self.da = da
        self.path = path
        self.kwargs = kwargs

    def on_train_begin(self, logs=None):
        self.clu = Clusterplot(*self.model.som_layer.map_size, honeycomb=True, region=get_region(self.da))

    def on_epoch_end(self, epoch, logs=None):
        for ax in self.clu.axes:
            ax.clear()

        decoded_centers = self.model.decoder(self.model.som_layer.prototypes)
        to_plot = [
            self.da.copy(data=decoded_center[..., 0]) for decoded_center in decoded_centers
        ]
        self.clu.add_contourf(to_plot, **self.kwargs)
        self.clu.fig.savefig(f'{self.path}/{epoch}.png')
        plt.show()
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['sigma'] = tf.keras.backend.get_value(self.model.sigma)