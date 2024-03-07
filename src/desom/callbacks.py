import io
import tensorflow as tf
from tensorflow import keras
from xarray import DataArray
import matplotlib.pyplot as plt
from jetstream_hugo.plots import Clusterplot
from jetstream_hugo.definitions import get_region
from IPython.display import display, clear_output
from desom.utils import rescale, descale


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
        
        
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


class PlotSOMCallback(keras.callbacks.Callback):
    def __init__(self, da: DataArray, Xmin, Xmax, **kwargs):
        super().__init__()
        if "time" in da.dims:
            da = da[0]
        self.da = da
        self.lon, self.lat = da.lon.values, da.lat.values
        self.Xmin, self.Xmax = Xmin, Xmax
        self.kwargs = kwargs


    def get_stuff_to_plot(self):
        try:
            decoded_centers = self.model.decoder(self.model.som_layer.prototypes)[..., 0]
        except AttributeError:
            protos = self.model.som_layer.prototypes
            decoded_centers = descale(protos.numpy(), self.Xmin, self.Xmax).reshape(protos.shape[0], *self.da.shape)
        to_plot = [
            self.da.copy(data=decoded_center) for decoded_center in decoded_centers
        ]
        to_plot = [tplt.where(tplt > 0, 0) for tplt in to_plot]
        return to_plot
    
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch > 0:
            return
        clu = Clusterplot(*self.model.som_layer.map_size, honeycomb=True, region=get_region(self.da))
        to_plot = self.get_stuff_to_plot()
        _ = clu.add_contourf(to_plot, **self.kwargs)
        display(clu.fig)    
    
    def on_epoch_end(self, epoch, logs=None):
        plt.clf()
        plt.close()
        to_plot = self.get_stuff_to_plot()
        clear_output(wait = True)
        clu = Clusterplot(*self.model.som_layer.map_size, honeycomb=True, region=get_region(self.da))
        _ = clu.add_contourf(to_plot, **self.kwargs)
        display(clu.fig) 
    