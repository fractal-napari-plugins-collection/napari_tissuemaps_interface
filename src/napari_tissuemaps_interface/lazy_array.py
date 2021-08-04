"""
Module containing a numpy-like array which supports lazy reading of tiled 2D-image data.
"""
import abc
import dask.array as da
import numpy as np


class LazyArray:
    """
    An abstract class of a numpy-like array which supports lazy reading of tiled 2D-image data.
    The class represents a custom array container which is compatible with the numpy API.
    For more details please refer to
    https://numpy.org/doc/stable/user/basics.dispatch.html#writing-custom-array-containers.

    The class is compatible with napari's image layer which expects a "numpy-like array" as
    input which supports indexing and can be converted to a numpy array via np.asarray.
    (ref: https://napari.org/tutorials/fundamentals/image.html#image-data-and-numpy-like-arrays)
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, shape, dtype, tile_size):
        """
        Initialization method.

        :param shape: The shape of the underlying array.
        :param dtype: The type of the underlying array.
        :param tile_size: The size of a single tile by which the image is divided.
        """
        assert len(shape) == 2
        self.shape = shape
        self.dtype = dtype
        self.tile_size = tile_size
        self.ndim = 2

    @property
    def size(self):
        """
        The number of elements in the array.
        """
        return self.shape[0] * self.shape[1]

    def __array__(self, dtype=None, **kwargs):
        # pylint: disable=W0613
        """
        Method used e.g. by numpy to obtain a standard numpy.ndarray.
        """
        return np.asarray(self[0:self.shape[0], 0:self.shape[1]])

    def __getitem__(self, idx):
        """
        Method which implements the support for basic slicing.
        It does not support field access nor advanced indexing.
        Moreover, the start and stop of a slice must be positive integers.

        This method is optimized for the napari viewer.
        napari calls self[:] for obtaining the shape, dtype and ndim attributes - not the data.
        To delay reading the underlying data this method does not return a numpy array
        but self when calling self[:].
        To access the underlying data napari calls np.asarray(self).
        """
        if not (
            isinstance(idx, slice) or
            (isinstance(idx, tuple) and all(isinstance(i, slice) for i in idx))
        ):
            raise ValueError("LazyArray only supports indexing by slices!")

        if (
            idx == slice(None, None, None) or
            idx == (slice(None, None, None), slice(None, None, None))
        ):
            return self

        if len(idx) != 2:
            raise Exception("Unsupported index!")
        (y_min, y_max), (x_min, x_max) = [(i.start, i.stop) for i in idx]
        y_off = y_min - (y_min % self.tile_size)
        x_off = x_min - (x_min % self.tile_size)

        assert (y_min >= 0) and (y_max >= 0) and (x_min >= 0) & (x_max >= 0)

        if y_max % self.tile_size == 0:
            max_y_tiles = (y_max // self.tile_size)
        else:
            max_y_tiles = (y_max // self.tile_size) + 1
        if x_max % self.tile_size == 0:
            max_x_tiles = (x_max // self.tile_size)
        else:
            max_x_tiles = (x_max // self.tile_size) + 1

        dask_arrays = []
        for y_tile in range(y_min // self.tile_size, max_y_tiles):
            row_tiles = []
            for x_tile in range(x_min // self.tile_size, max_x_tiles):
                row_tiles.append(
                    da.from_delayed(
                        self.read_tile(y_tile, x_tile),
                        shape=(self.tile_size, self.tile_size), dtype=np.uint8
                    )
                )
            dask_arrays.append(row_tiles)

        y_max = min(y_max, self.shape[0])
        x_max = min(x_max, self.shape[1])
        return da.block(dask_arrays)[y_min-y_off:y_max-y_off, x_min-x_off:x_max-x_off]

    @abc.abstractmethod
    def read_tile(self, y_tile, x_tile):
        """
        Abstract method which reads a tile at the position (y_tile, x_tile).
        """
        return
