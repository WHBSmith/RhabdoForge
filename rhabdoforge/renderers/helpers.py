from typing import Tuple, Sequence, Optional, Union, Any
import os
import tempfile
from pathlib import Path
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes

from rhabdoforge.compound_eyes import Model
from rhabdoforge.geometry.circular import wrap_angle


class HistoryRecorder:
    """
    Disk-backed, append-only store for a stream of fixed-shape frames.
    """

    def __init__(self,
             frame_shape: Tuple[int, ...],
             dtype: npt.DTypeLike = np.float32,
             path: Optional[Union[str, Path]] = None,
             init_capacity: int = 256
         ):

        self._frame_shape = tuple(frame_shape)
        self._dtype = np.dtype(dtype)
        self._frame_nbytes = int(np.prod(self._frame_shape)) * self._dtype.itemsize

        self._owns_file = path is None
        if path is None:
            fd, path = tempfile.mkstemp(suffix='.rfhist')
            os.close(fd)

        self._path = Path(path)

        self._count = 0
        self._capacity = 0
        self._mmap: Optional[np.memmap] = None
        self._grow(max(1, init_capacity))

    def _grow(self, min_capacity: int) -> None:
        new_capacity = max(min_capacity, self._capacity * 2)

        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None

        with open(self._path, 'r+b' if self._path.exists() else 'w+b') as f:
            f.truncate(new_capacity * self._frame_nbytes)

        self._mmap = np.memmap(
            self._path,
           dtype=self._dtype,
           mode='r+',
           shape=(new_capacity, *self._frame_shape)
       )

        self._capacity = new_capacity

    def extend(self, chunk: np.ndarray) -> None:
        """Appends one frame (shape =frame_shape) or a batch of frames (shape = (T, *frame_shape))."""

        chunk = np.asarray(chunk, dtype=self._dtype)
        if chunk.shape == self._frame_shape:
            chunk = chunk[np.newaxis, ...]
        elif chunk.ndim != len(self._frame_shape) + 1 or chunk.shape[1:] != self._frame_shape:
            raise ValueError(f'Chunk shape {chunk.shape} incompatible with frame shape {self._frame_shape}')

        n = chunk.shape[0]
        if self._count + n > self._capacity:
            self._grow(self._count + n)

        self._mmap[self._count:self._count + n] = chunk
        self._count += n

    @property
    def array(self) -> np.ndarray:
        """Read-only memmaped view over the recorded frames, shape (len(self), *frame_shape)."""
        return self._mmap[:self._count]

    def __len__(self) -> int:
        return self._count

    def clear(self) -> None:
        """Reset frame count (backing file/capacity is kept for reuse)."""
        self._count = 0

    def close(self) -> None:
        """Releases the memmap, delete the temp backing file (custom paths are not deleted)."""
        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None
        if self._owns_file and self._path.exists():
            try:
                self._path.unlink()
            except OSError:
                pass

    def __del__(self):
        self.close()


class SignalView:
    """
    Thin wrapper around any array ending in (R, G, B, gain).
    (allows channel extraction chaining, e.g. 'output.per_cartridge.radiance')
    """
    __slots__ = ('_data',)

    def __init__(self, data: np.ndarray):
        self._data = data

    def __repr__(self) -> str:
        return f'SignalView(shape={self.shape})'

    def __len__(self) -> int:
        return self._data.shape[0]

    def __bool__(self) -> bool:
        return self._data is not None and self._data.size > 0

    def copy(self) -> 'SignalView':
        """Returns a deep copy of the view and its data."""
        return SignalView(self._data.copy())

    def __eq__(self, other: Any) -> Union[bool, np.ndarray]:
        other_data = other._data if isinstance(other, SignalView) else other
        return self._data == other_data

    def __array__(self, dtype: npt.DTypeLike = None, copy: Optional[bool] = None) -> np.ndarray:
        if dtype is not None or copy is not None:
            return np.array(object=self._data, dtype=dtype, copy=copy)
        return self._data

    def __getitem__(self, idx: Any) -> Union['SignalView', np.ndarray]:
        res = self._data[idx]
        if isinstance(res, np.ndarray) and res.ndim >= 1 and res.shape[-1] == 4:
            return SignalView(res)
        return res

    @property
    def data(self) -> np.ndarray:
        """The raw array."""
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._data.shape

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def ndim(self) -> int:
        return self._data.ndim

    # Channel helpers

    @property
    def colours(self) -> np.ndarray:
        """The adapted spectral response (Rhabdomere output)."""
        return self._data[..., :3]

    @property
    def adaptation(self) -> np.ndarray:
        """
        The adaptation state (gain factor) of the rhabdomeres.
        This is the value calculated by the Naka-Rushton equations (0.0 to 1.0+).
        """
        return self._data[..., 3]

    @property
    def gain(self) -> np.ndarray:
        """Alias to adaptation"""
        return self.adaptation

    @property
    def raw_radiance(self) -> np.ndarray:
        """
        The light intensity hitting the eye before adaptation.
        Recovered by 'un-baking' the adaptation factor.
        """
        return np.mean(self.colours, axis=-1) / (self.adaptation + 1e-6)

    @property
    def radiance(self) -> np.ndarray:
        """
        The mean intensity of the adapted signal (mean of the colour channels).
        """
        return np.mean(self.colours, axis=-1)

    def normalized(self) -> 'SignalView':
        """Returns the signal scaled to [0, 1] based on its own min/max."""
        d = self._data
        low, high = d.min(), d.max()
        return SignalView((d - low) / (high - low + 1e-8))


class VisualOutput(SignalView):
    """
    Per-rhabdomere output array with various biological pathway mappings conveniences.

    The renderers return a float array where the last axis is (R, G, B, radiance).
    This class supports both single snapshots (shape: N, 4) and timeseries (shape: T, N, 4).

    Layouts / pathways:
        .per_ommatidium    -> (..., N, R, 4) Physical ommatidia grouping.
        .per_cartridge     -> (..., N, R, 4) Neural superposition grouping.
        .per_rhabdomere(i) -> (..., N, 4)    Specific rhabdomere type across all cartridges.
        .peripheral_signal -> (..., N, 4)    Pooled R1-R6 (LMC pathway for motion).
        .central_signal    -> (..., N, 4)    Central R7/R8 (Medulla colour pathway).
        .lmc_input         -> (..., N, 4)    Alias for peripheral_signal.
        .pale_input        -> (..., N, 4)    Alias for central_signal.

    Signal analysis:
        .colours      -> (..., 3) The adapted spectral response.
        .adaptation   -> (..., )  The gain factor (state of the biological system).
        .radiance     -> (..., )  Adapted intensity (mean of adapted RGB).
        .raw_radiance -> (..., )  Physical light intensity recovered by 'un-baking'
                                  the adaptation factor.
    """
    __slots__ = ('_model', '_coords', '_is_time_series', '_depth', '_depth_stats', '_raw_depth')

    def __init__(self, data: Any, model: 'Model', coords: str = 'rhabdomeres', is_time_series: Optional[bool] = None,
                 depth: Optional[np.ndarray] = None,
                 depth_stats: Optional[dict] = None,
                 raw_depth: Optional[np.ndarray] = None):
        super().__init__(np.asarray(data))
        self._model = model
        self._depth = None if depth is None else np.asarray(depth)
        self._depth_stats = depth_stats
        self._raw_depth = raw_depth
        self._coords = coords

        if self._data.shape[-2] % model.shape[1] != 0 and coords == 'rhabdomeres':
            raise ValueError(f'data length {self._data.shape[-2]} not divisible by R={model.shape[1]}')

        if is_time_series is None:
            self._is_time_series = (self.ndim == 3 and self._coords == 'rhabdomeres') or \
                                   (self.ndim == 4 and self._coords == 'ommatidia') or \
                                   (self.ndim == 3 and self._coords == 'ommatidia' and self._data.shape[-2] ==
                                    model.shape[0])
        else:
            self._is_time_series = is_time_series

    @property
    def depth(self) -> Optional[np.ndarray]:
        """Mean valid primary-ray depth in metres, if requested by the renderer."""
        return self._depth

    @property
    def depth_stats(self) -> Optional[dict]:
        """Reduced primary-ray statistics, if requested by the renderer."""
        return self._depth_stats

    @property
    def raw_depth(self) -> Optional[np.ndarray]:
        """Per-ray primary depths for diagnostic renders only."""
        return self._raw_depth

    def __repr__(self) -> str:
        t_str = f'Time={self.shape[0]}, ' if self._is_time_series else ''
        if self._coords == 'rhabdomeres':
            g_str = ''
            N = self.shape[-2] // self._model.shape[1]
        else:
            g_str = ', group=omm/cart'
            N = self._model.shape[0]
        return f'VisualOutput({t_str}N={N}, R={self._model.shape[1]}{g_str})'

    def __getitem__(self, idx: Any) -> Union['VisualOutput', SignalView, np.ndarray]:
        res = self._data[idx]
        if isinstance(res, np.ndarray) and res.ndim >= 1 and res.shape[-1] == 4:
            N = self._model.shape[0]
            R = self._model.shape[1]

            # Pattern 1: (..., N*R, 4)
            if res.shape[-2] == N * R:
                return VisualOutput(res, self._model, coords='rhabdomeres', is_time_series=res.ndim == 3)
            # Pattern 2: (..., N, R, 4)
            elif res.ndim >= 2 and res.shape[-3:-1] == (N, R):
                return VisualOutput(res, self._model, coords='ommatidia', is_time_series=res.ndim == 4)
            # Pattern 3: (..., N, 4) -> Pooled pathway
            elif res.shape[-2] == N:
                return VisualOutput(res, self._model, coords='ommatidia', is_time_series=res.ndim == 3)

            return SignalView(res)
        return res

    def copy(self) -> 'VisualOutput':
        """Returns a deep copy of the visual output, preserving the model."""
        return VisualOutput(self._data.copy(), self._model, self._coords, self._is_time_series)

    def _memmap_base(self) -> Optional[np.memmap]:
        arr = self._data
        while arr is not None:
            if isinstance(arr, np.memmap):
                return arr
            arr = getattr(arr, 'base', None)
        return None

    @property
    def is_memmap(self) -> bool:
        """True if the data is backed by a memory-mapped file."""
        return self._memmap_base() is not None

    @property
    def file_path(self) -> Optional[Path]:
        """Backing file path if memmap-backed, else None."""
        base = self._memmap_base()
        return Path(base.filename) if base is not None and base.filename else None

    @classmethod
    def from_history(cls, history: Sequence['VisualOutput'] | Sequence[np.ndarray],
                     model: Optional['Model'] = None) -> 'VisualOutput':
        """
        Stacks or concatenates multiple inputs into a single timeseries VisualOutput.
        """
        if not history:
            raise ValueError('History is empty')

        arrays = [np.asarray(item) for item in history]
        normalized = [a[np.newaxis, ...] if a.ndim == 2 else a for a in arrays]

        # find the model from the first VisualOutput in the list
        found_model = model
        if found_model is None:
            for item in history:
                if isinstance(item, VisualOutput):
                    found_model = item._model
                    break

        if found_model is None:
            raise ValueError('A Model must be provided if the history consists only of raw arrays.')

        combined_data = np.concatenate(normalized, axis=0)

        return cls(combined_data, found_model, coords='rhabdomeres', is_time_series=True)

    @property
    def latest(self) -> 'VisualOutput':
        """Returns the last frame as a 2D VisualOutput."""
        if self._is_time_series:
            return VisualOutput(self._data[-1], self._model, self._coords, is_time_series=False)
        return self

    @property
    def per_ommatidium(self) -> 'VisualOutput':
        """Returns (..., N, R, 4) array of all rhabdomere outputs, per ommatidium."""
        if self._coords == 'ommatidia':
            if self._data.ndim >= 3 and self._data.shape[-3:-1] == (self._model.shape[0], self._model.shape[1]):
                return self
            raise ValueError('Data is already pooled/flattened and cannot be unpacked to ommatidia.')

        N, R = self._model.shape[0], self._model.shape[1]
        prefix = self.shape[:-2]
        data = self._data.reshape(*prefix, N, R, 4)
        return VisualOutput(data, self._model, coords='ommatidia', is_time_series=self._is_time_series)

    @property
    def per_cartridge(self) -> 'VisualOutput':
        """Returns (..., N, R, 4) array of all rhabdomere outputs, per cartridge."""
        if not self._model.neural_superposition:
            return self.per_ommatidium

        # convert back to a flat (..., N*R, 4) representation first so indices apply correctly
        if self._coords == 'rhabdomeres':
            flat_data = self._data
        elif self._coords == 'ommatidia' and self._data.ndim >= 3 and self._data.shape[-3:-1] == (self._model.shape[0],
                                                                                                  self._model.shape[1]):
            prefix = self._data.shape[:-3]
            flat_data = self._data.reshape(*prefix, self._model.size, 4)
        else:
            raise ValueError("Data cannot be grouped into cartridges from its current pooled shape.")

        idx = self._model.cartridge_indices
        data = np.take(flat_data, idx, axis=-2)
        return VisualOutput(data, self._model, coords='ommatidia', is_time_series=self._is_time_series)

    def per_rhabdomere(self, index: int) -> 'VisualOutput':
        """Returns (..., N, 4) array for a specific rhabdomere index (e.g. 0 for R1)."""
        data = self.per_cartridge.data[..., index, :]
        return VisualOutput(data, self._model, coords='ommatidia', is_time_series=self._is_time_series)

    @property
    def peripheral_signal(self) -> 'VisualOutput':
        """The pooled response of all peripheral rhabdomeres (LMC-pathway)."""
        if self._model.shape[1] == 1:
            return self.per_ommatidium.per_rhabdomere(0)

        periph_idx = getattr(self._model.bundle, 'peripheral_indices', [])

        pc = self.per_cartridge.data  # (..., N, R, 4)
        if len(periph_idx) == 0:
            data = pc[..., 0, :]
        else:
            data = np.mean(pc[..., periph_idx, :], axis=-2)

        return VisualOutput(data, self._model, coords='ommatidia', is_time_series=self._is_time_series)

    @property
    def central_signal(self) -> 'VisualOutput':
        """The response of the central rhabdomere (Medulla colour pathway)."""
        c_idx = getattr(self._model.bundle, 'center_index', 0)
        return self.per_rhabdomere(c_idx)

    @property
    def lmc_input(self) -> 'VisualOutput':
        """Alias to peripheral_signal"""
        return self.peripheral_signal

    @property
    def pale_input(self) -> 'VisualOutput':
        """Alias to central_signal"""
        return self.central_signal

    # TODO: The str-based modes selection is crap, need to replace

    def plot(self,
            ax: Optional['Axes'] = None,
            pathway: str = 'all',
            projection: str = 'equirectangular',
            edgecolor: Optional[str] = None, # None for seamless, or a colour ('#000000', 'black', 'k', etc) for outlines
            gamma: float = 2.2
        ) -> 'Axes':
        """
        Displays the visual output as a gapless Voronoi tessellation.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from scipy.spatial import Voronoi

        projection = str(projection).lower() if projection else 'equirectangular'
        pathway = str(pathway).lower() if pathway else 'all'

        if ax is None:
            if 'equirect' in projection:
                fig, ax = plt.subplots(figsize=(10, 5))
            else:
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111, projection=projection)

        if pathway != 'all':
            if self._coords != 'rhabdomeres':
                raise ValueError(f"Cannot apply pathway '{pathway}' on data that is already grouped.")
            match pathway:
                case 'peripheral' | 'periph':
                    view = self.peripheral_signal
                case 'central':
                    view = self.central_signal
                case 'ommatidium' | 'omm':
                    view = self.per_ommatidium
                case 'cartridge':
                    view = self.per_cartridge
                case _:
                    raise ValueError(
                        "Invalid pathway, must be 'all', 'peripheral', 'central', 'ommatidium', or 'cartridge'")
        else:
            view = self

        # Get latest data
        data = view.data[-1] if view._is_time_series else view.data

        N = self._model.shape[0]
        R = self._model.shape[1]

        # determine spatial dim and coordinates
        if data.ndim == 3 and data.shape[-3:-1] == (N, R):
            # (N, R, 4) -> grouped for plotting
            plot_data = np.mean(data, axis=-2)
            az, el = self._model.ommatidia.azimuth, self._model.ommatidia.elevation
        elif data.ndim == 2 and data.shape[-2] == N:
            # (N, 4)
            plot_data = data
            az, el = self._model.ommatidia.azimuth, self._model.ommatidia.elevation
        elif data.ndim == 2 and data.shape[-2] == N * R:
            # (N*R, 4)
            plot_data = data
            az, el = self._model.rhabdomeres.azimuth, self._model.rhabdomeres.elevation
        else:
            raise ValueError(f'Unsupported shape for plotting: {data.shape}')

        rgb = np.clip(plot_data[..., :3], 0.0, 1.0).copy()

        # Apply Gamma Correction
        if gamma and gamma != 1.0:
            rgb = np.power(rgb, 1.0 / gamma)

        # Peripheral is greyscale
        if 'periph' in pathway:
            rgb = np.repeat(np.mean(rgb, axis=-1, keepdims=True), 3, axis=-1)

        # Voronoi cells on the unwrapped cylinder
        pts = np.column_stack((az, el))
        pts_left = np.column_stack((az - 2 * np.pi, el))
        pts_right = np.column_stack((az + 2 * np.pi + 1e-6, el))
        cap_x = np.linspace(-2 * np.pi, 2 * np.pi, max(100, len(az) // 10))
        pts_top = np.column_stack((cap_x, np.full_like(cap_x, np.pi / 2 + 0.5)))
        pts_bot = np.column_stack((cap_x, np.full_like(cap_x, -np.pi / 2 - 0.5)))

        vor = Voronoi(np.vstack((pts, pts_left, pts_right, pts_top, pts_bot)))

        polygons = []
        for i in range(len(pts)):
            polygons.append(vor.vertices[vor.regions[vor.point_region[i]]])

        if 'equirect' in projection:
            polygons = [np.degrees(p) for p in polygons]
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Elevation (deg)')
            ax.set_aspect('equal', adjustable='box')
        else:
            for p in polygons:
                p[:, 0] = wrap_angle(p[:, 0])
            ax.grid(True, alpha=0.3)

        # Anti-aliasing workaround: 'face' draws a mini border of the polygon's colour over the gaps left by matplotlib
        edgecolor = str(edgecolor).lower() or 'none'
        edge_c = 'face' if edgecolor == 'none' else edgecolor
        pc = PolyCollection(polygons, facecolors=rgb, edgecolors=edge_c, linewidths=0.5, antialiaseds=True)
        ax.add_collection(pc)

        # TODO: Add title to plot

        return ax

    def plot_timeseries(self,
            ax: Optional['Axes'] = None,
            pathway: str = 'peripheral',
            max_items: int = 100,
            sort_by: str = 'azimuth',
            gamma: float = 2.2
        ) -> 'Axes':
        """
        Plots a spatio-temporal heatmap of the visual output over time.
        """
        import matplotlib.pyplot as plt

        sort_by = str(sort_by).lower() if sort_by else 'azimuth'
        pathway = str(pathway).lower() if pathway else 'all'

        if not self._is_time_series:
            raise ValueError('This visualisation requires a VisualOutput containing multiple timesteps.')
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, min(10, max_items * 0.1))))

        if pathway != 'all':
            if self._coords != 'rhabdomeres':
                raise ValueError('Cannot derive pathways from data already grouped by ommatidia.')

            match pathway:
                case 'peripheral' | 'periph':
                    view = self.peripheral_signal
                case 'central':
                    view = self.central_signal
                case 'ommatidium' | 'omm':
                    view = self.per_ommatidium
                case 'cartridge':
                    view = self.per_cartridge
                case _:
                    raise ValueError(f'Invalid pathway: {pathway}')
        else:
            view = self

        data = view.data
        N = self._model.shape[0]
        R = self._model.shape[1]

        if data.ndim == 4 and data.shape[-3:-1] == (N, R):
            plot_data = np.mean(data, axis=-2)
            az = self._model.ommatidia.azimuth
            ylabel_default = 'Grouped items'
        elif data.ndim == 3 and data.shape[-2] == N:
            plot_data = data
            az = self._model.ommatidia.azimuth
            ylabel_default = 'Pathways'
        elif data.ndim == 3 and data.shape[-2] == N * R:
            plot_data = data
            az = self._model.rhabdomeres.azimuth
            ylabel_default = 'Rhabdomeres'
        else:
            raise ValueError(f'Unsupported shape for timeseries: {data.shape}')

        plot_data = np.clip(plot_data[..., :3], 0.0, 1.0)

        # Apply Gamma Correction
        if gamma and gamma != 1.0:
            plot_data = np.power(plot_data, 1.0 / gamma)

        if 'periph' in pathway:
            plot_data = np.repeat(np.mean(plot_data, axis=-1, keepdims=True), 3, axis=-1)

        # Transpose to (space, time, RGB) for imshow
        plot_data = np.transpose(plot_data, (1, 0, 2))
        if 'az' in sort_by:
            plot_data = plot_data[np.argsort(az)]

        # downsample (spatially) if needed
        S, T, C = plot_data.shape
        if S > max_items:
            block_size = S // max_items
            plot_data = plot_data[:block_size * max_items].reshape(max_items, block_size, T, C).mean(axis=1)

        ax.imshow(plot_data, aspect='auto', origin='lower', interpolation='none')
        ax.set_xlabel('Time step')

        ylabel = 'Items'
        if pathway == 'all':
            ylabel = ylabel_default
        elif pathway in ('cartridge', 'peripheral', 'periph', 'central'):
            ylabel = 'Cartridges'
        elif 'omm' in pathway:
            ylabel = 'Ommatidia'

        ax.set_ylabel(ylabel + (' (sorted Left to Right)' if 'az' in sort_by else ''))

        return ax

    def plot_curves(self,
            ax: Optional['Axes'] = None,
            pathway: str = 'peripheral',
            index: Union[int, str] = 'mean',
            mode: str = 'rgb',
        ) -> 'Axes':
        """
        Plots the temporal signal (RGB or luminance) as lines over time.

        Args:
            - ax: Optional, Matplotlib axes.
            - pathway: The neural pathway to plot ('peripheral', 'central', 'all', etc.)
            - index: The index of the ommatidium/rhabdomere to plot. Use 'mean' to plot the average across the whole slice.
            - mode: 'rgb' to plot individual color channels, 'luminance' for a single intensity line.
        """
        import matplotlib.pyplot as plt

        if not self._is_time_series:
            raise ValueError('This visualisation requires a VisualOutput containing multiple timesteps.')

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))

        pathway = str(pathway).lower()
        if pathway != 'all':
            if self._coords != 'rhabdomeres':
                raise ValueError('Cannot derive pathways from data already grouped.')
            match pathway:
                case 'peripheral' | 'periph':
                    view = self.peripheral_signal
                case 'central':
                    view = self.central_signal
                case 'ommatidium' | 'omm':
                    view = self.per_ommatidium
                case 'cartridge':
                    view = self.per_cartridge
                case _:
                    raise ValueError(f'Invalid pathway: {pathway}')
        else:
            view = self

        data = view.data

        # Spatial reduction: specific index or global mean
        if index == 'mean':
            signal = np.mean(data, axis=1)
        else:
            signal = data[:, int(index)]

        # If data is still grouped by Rhabdomeres, mean across R
        if signal.ndim == 3:
            signal = np.mean(signal, axis=1)

        rgb = np.clip(signal[..., :3], 0.0, 1.0)

        time_steps = np.arange(len(rgb))

        if mode.lower() == 'rgb':
            r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

            ax.plot(time_steps, r, color='red', label='Red', alpha=0.8)
            ax.plot(time_steps, g, color='green', label='Green', alpha=0.8)
            ax.plot(time_steps, b, color='blue', label='Blue', alpha=0.8)
        else:
            # Luminance mode
            lum = np.mean(rgb, axis=-1)
            ax.plot(time_steps, lum, color='black', label='Luminance')

        ax.set_xlabel('Time step')
        ax.set_ylabel('Signal amplitude')
        title = f'Temporal signal ({pathway})'
        if index == 'mean':
            title += ' - Global mean'
        else:
            title += f' - Unit {index}'
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax
