"""
Consumer-agnostic retinotopic resampling between two samplings of the visual field.

A 'SamplingGrid' is an ordered set of sample points living in a 2D "retinal plane",
optionally carrying their world directions, the frame that plane was projected in,
and the global source indices they were built from.

A 'DiscreteResampler' maps one sensor grid onto another by nearest-neighbour (or inverse-distance) matching
in a shared, normalised retinal plane, and re-indexes a per-source signal into target order.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional, Sequence, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from rhabdoforge.geometry.spherical import cartesian_to_spherical

from rhabdoforge.geometry.spherical import sphere_to_stereo
from rhabdoforge.utils import infer_name, norm_rms

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from rhabdoforge.types import ReferenceFrame
    from rhabdoforge.compound_eyes.views import BaseView
    from rhabdoforge.compound_eyes.helpers import SignalView


logger = logging.getLogger(__name__)


@dataclass(frozen=True, eq=False)  # eq=False: fields are ndarrays (array equality isn't scalar)
class SamplingGrid:
    """
    An ordered set of visual-field sample points in a 2D retinal plane.

        positions2d: (M, 2) retinal-plane layout, in lattice order (raw, un-normalised).
        directions : (M, 3) world directions, if meaningful (None for abstract targets).
        frame      : (forward, right, up) the plane was projected in, if applicable.
        index      : (M,) global source indices these were built from (model eyes), else None.
    """
    positions2d: np.ndarray
    directions: Optional[np.ndarray] = None
    frame: Optional['ReferenceFrame'] = None
    index: Optional[np.ndarray] = None
    name: str = ''

    @property
    def size(self) -> int:
        return int(self.positions2d.shape[0])

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        tag = f' {self.name!r}' if self.name else ''
        return f'{self.__class__.__name__}({self.size} pts{tag})'

    @classmethod
    def from_positions2d(cls,
            xy: ArrayLike,
            index: Optional[Sequence[int]] = None,
            name: str = '',
            flip_x: bool = False,
            flip_y: bool = False,
            swap_xy: bool = False,
    ) -> 'SamplingGrid':
        """
        Build from an explicit 2D layout.
        The flip / swap knobs are the target-side handedness convention.
        """
        xy = np.array(xy, dtype=np.float64, copy=True).reshape(-1, 2)
        if swap_xy:
            xy = xy[:, ::-1].copy()
        if flip_x:
            xy[:, 0] *= -1.0
        if flip_y:
            xy[:, 1] *= -1.0
        idx = None if index is None else np.asarray(index, dtype=np.intp).reshape(-1)
        return cls(positions2d=xy, index=idx, name=name)

    @classmethod
    def from_directions(cls,
            directions: ArrayLike,
            index: Optional[Sequence[int]] = None,
            name: str = ''
        ) -> 'SamplingGrid':
        """
        Build from world directions via stereographic projection about their mean.
        """
        dirs = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
        xy, fwd, right, up = sphere_to_stereo(dirs)
        lat = cls.from_positions2d(xy, index=index, name=name)
        return replace(lat, directions=dirs, frame=(fwd, right, up))

    @classmethod
    def from_model(cls, view: 'BaseView', name: str = '') -> 'SamplingGrid':
        """
        Source sensor grid from any ommatidium-anchored view (a model, an eye, a slice, a spatial query result...).
        """
        idx = np.asarray(view.indices, dtype=np.intp)

        if not name:
            name = infer_name(view, depth=2, fallback=type(view).__name__)

        return cls.from_directions(view.directions, index=idx, name=name)


class Resampler(ABC):
    """
    Shared interface for objects that resample the visual field onto a fixed target
    sampling (a SamplingGrid). Built once against a `target`, then called per frame.

    Subclasses differ only in what the incoming 'field' is:
        - DiscreteResampler: the field is a per-element signal living on a *discrete*
                              source SamplingGrid; sampling is a gather (+ optional
                              distance weighting) between the two samplings.
        - ContinuousResampler: the field is a *continuous* equirectangular image;
                              sampling is interpolation at each target direction.

    Contract: `__call__` returns the target inputs with the spatial axis matching the
    location of the original source spatial axis (e.g. replacing length M with length T).
    """

    target: 'SamplingGrid'  # the destination sampling; set by each subclass

    @property
    def size(self) -> int:
        """T, the number of target sample points."""
        return self.target.size

    def __len__(self) -> int:
        return self.size

    @abstractmethod
    def __call__(self, payload: ArrayLike, **kwargs) -> np.ndarray:
        """Sample `payload` onto `target`. Returns (..., T) or (..., T, C)."""
        ...


@dataclass(eq=False)
class DiscreteResampler(Resampler):
    """
    A gather from a 'source' sensor grid onto a 'target' sensor grid.
    Supports both topological matching (shape-overlay) and exact 3D angular matching.

    Modes (method):
        - 'topology':
            Both sensor grids are projected to 2D, centred, and uniformly scaled to
            a unit RMS radius. Target points then pull from their nearest source points
            in this normalised space.
            Use this for: Mapping a biological eye (curved, physical) to an abstract
            target (like a flat FlyVis hex grid) or overlapping eyes with vastly
            different FOVs. It perfectly overlays shapes while ignoring physical scale.

        - '3d' (default):
            Finds the nearest neighbours using the raw 3D optical axes (chord distance).
            Zero scaling or projection distortion is applied.
            Use this for: Model-to-model resampling (e.g. mapping a high-res eye
            onto a low-res model of the same species). Both models must
            share the same 3D world coordinate frame (they usually do).

    Weighting (sigma):
        When called, providing a `sigma` will apply a Gaussian weight to the neighbours.
        If omitted, inverse-distance weighting is used. For exact biological precision
        (preserving the renderer's optical blur), use k=1. For anti-aliasing between
        mismatched grids, use k>1 with a small sigma (e.g. half the lattice spacing).
    """
    source: 'SamplingGrid'
    target: 'SamplingGrid'
    k: int = 1
    method: str = '3d'  # 'topology' or '3d'

    indices: np.ndarray = field(init=False)       # (T,) or (T, k) into source order
    distances: np.ndarray = field(init=False)     # (T,) or (T, k) normalised distances

    def __post_init__(self):
        if self.method == '3d' and self.source.directions is not None and self.target.directions is not None:
            # Match using actual 3D optical axes (chord distances)
            self.distances, self.indices = cKDTree(self.source.directions).query(self.target.directions, k=self.k)
        else:
            if self.method == '3d':
                logger.warning("Method '3d' requested but directions are missing. Falling back to 'topology'.")
                self.method = 'topology'

            # Match using normalised 2D shape overlap
            src = norm_rms(self.source.positions2d)
            tgt = norm_rms(self.target.positions2d)
            self.distances, self.indices = cKDTree(src).query(tgt, k=self.k)

    def __repr__(self) -> str:
        src_name = self.source.name or "source"
        tgt_name = self.target.name or "target"
        return f"<{self.__class__.__name__} '{self.method}': {src_name} -> {tgt_name} (k={self.k})>"

    def __call__(self,
            values: Union['SignalView', ArrayLike],
            max_dist: Optional[float] = None,
            fill: float = 0.0,
            sigma: Optional[float] = None
        ) -> np.ndarray:
        """
        Re-index a per-source signal into target order.

        Args:
            - values: Array or VisualOutput/SignalView.
                      Shape must end in (M,) or (M, C) where M is the source sensor grid size.
                      If the source sensor grid has a global 'index', M can be the full model size.
            - max_dist: optional float, cut in normalised units, target points whose nearest
                        source is farther than this are set to 'fill' (e.g. blind spot).
            - fill: float, fill value for target points whose nearest source is farther than 'max_dist'
            - sigma: If provided, applies a Gaussian weight exp(-d^2 / 2*sigma^2).
                        If None, falls back to inverse-distance weighting (1/d).

        Returns:
            An array with the spatial dimension replaced by T (target size), e.g., (..., T) or (..., T, C).
        """
        v = np.asarray(getattr(values, 'data', values), dtype=np.float32)
        M = self.source.size

        # Figure out whether spatial dimension is last or second-to-last
        spatial_axis = -1
        if v.ndim >= 2:
            if v.shape[-2] == M and v.shape[-1] != M:
                spatial_axis = -2
            elif v.shape[-1] == M:
                spatial_axis = -1
            else:
                if self.source.index is not None:
                    max_idx = np.max(self.source.index)
                    if v.shape[-2] > max_idx and v.shape[-1] in (1, 3, 4):
                        spatial_axis = -2
                    elif v.shape[-1] > max_idx:
                        spatial_axis = -1
                    else:
                        raise ValueError(
                            f'Expected spatial dimension >= {max_idx + 1} '
                            f'in last two axes, but found shape {v.shape}.'
                        )
                else:
                    raise ValueError(
                        f'Expected spatial dimension of size {M} in last two axes, '
                        f'but found shape {v.shape}.'
                    )
        else:
            if v.shape[0] != M:
                if self.source.index is not None:
                    max_idx = np.max(self.source.index)
                    if v.shape[0] <= max_idx:
                        raise ValueError(f'Expected spatial dimension >= {max_idx + 1}, found {v.shape[0]}')
                else:
                    raise ValueError(f'Expected spatial dimension of size {M}, found {v.shape[0]}')

        # Reduce to just the FOV we're interested in before sampling
        if self.source.index is not None and v.shape[spatial_axis] != M:
            v = np.take(v, self.source.index, axis=spatial_axis)

        if self.indices.ndim == 2:
            if sigma is not None:
                # Gaussian weighting
                w = np.exp(- (self.distances ** 2) / (2 * sigma ** 2))
            else:
                # Inverse-distance weighting
                w = 1.0 / (self.distances + 1e-6)

            w_sum = w.sum(axis=1, keepdims=True)
            w = np.divide(w, w_sum, out=np.zeros_like(w), where=w_sum > 1e-12)

            v_sampled = np.take(v, self.indices, axis=spatial_axis)
            if spatial_axis == -2:
                out = np.sum(v_sampled * w[..., np.newaxis], axis=-2)
            else:
                out = np.sum(v_sampled * w, axis=-1)
            d0 = self.distances[:, 0]
        else:
            # Nearest neighbour (k=1)
            out = np.take(v, self.indices, axis=spatial_axis)
            d0 = self.distances

        if max_dist is not None:
            mask = d0 > max_dist
            if spatial_axis == -2:
                mask = mask[..., np.newaxis]
            out = np.where(mask, fill, out)

        return out

    @property
    def coverage(self) -> float:
        """
        Fraction of source elements feeding at least one target point.
        """
        return float(np.unique(self.indices).size) / max(self.source.size, 1)

    @property
    def target_coords(self) -> np.ndarray:
        """
        Raw target layout (for eyeballing orientation / flips).
        """
        return self.target.positions2d


class ContinuousResampler(Resampler):
    """
    Samples an equirectangular image/video frame into a target set of
    directions (e.g. the compound eye's ommatidia).

    Here the 'source' is the continuous panorama and the `target` SamplingGrid is the
    ommatidia sampling. That grid's 2D layout is simply the (azimuth, elevation) plane,
    which *is* the equirectangular parameterisation, so no stereographic projection
    needed, and full 360-degree fields are fine (unlike the topology mapper).
    """

    def __init__(self, directions: ArrayLike):
        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)

        # Convert 3D optical axes to spherical azimuth (-pi..pi) and elevation (-pi/2..pi/2)
        self.az, self.el = cartesian_to_spherical(dirs)

        # Target sampling, directly in the (az, el) plane
        self.target = SamplingGrid(
            positions2d=np.column_stack([self.az, self.el]).astype(np.float64),
            directions=dirs.astype(np.float64),
            name='panorama',
        )

    def __repr__(self) -> str:
        tgt_name = self.target.name or 'panorama_target'
        return f"<{self.__class__.__name__}: Equirectangular -> {tgt_name} (size={self.size})>"

    @property
    def directions(self) -> Optional[np.ndarray]:
        """World optical axes of the target sampling (for convenience)."""
        return self.target.directions

    @classmethod
    def from_model(cls, view: 'BaseView') -> 'ContinuousResampler':
        """
        Build directly from any ommatidium-anchored view (mirrors SamplingGrid.from_model),
        pulling its world optical axes.
        """
        return cls(view.directions)

    def _coords(self, H: int, W: int) -> np.ndarray:
        """(2, T) row/col image coordinates per target direction, for map_coordinates."""
        # Azimuth [-pi, pi] -> x pixel [0, W-1]
        u = ((self.az + np.pi) / (2 * np.pi)) * (W - 1)
        # Elevation [pi/2, -pi/2] -> y pixel [0, H-1] (image y points down)
        v = ((np.pi / 2 - self.el) / np.pi) * (H - 1)
        return np.vstack((v, u))

    def sample(self, image: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Image-native sampling.

        image: (H, W, C) or (H, W) array (e.g. a video frame).
        order: 1 for bilinear interpolation, 0 for nearest neighbour.
        Returns: (T,) for greyscale or (T, C) for colour (points x channels).
        """
        H, W = image.shape[:2]
        coords = self._coords(H, W)

        if image.ndim == 3:
            C = image.shape[2]
            out = np.zeros((self.target.size, C), dtype=image.dtype)
            for c in range(C):
                out[:, c] = map_coordinates(image[..., c], coords, order=order, mode='wrap')
            return out

        return map_coordinates(image, coords, order=order, mode='wrap')

    def __call__(self, image: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Unified Resampler call.
        """
        return self.sample(image, order=order)


##

# The FlyVis implementation

def flyvis_hex_uv(extent: int = 15) -> np.ndarray:
    """
    (u, v) axial coordinates of a flyvis hex eye, for the given 'extent'.
    This is in FlyVis raster order: (u outer, v inner), identical to FlyVis' BoxEye._receptor_centers

    n_hexals = 3 * extent * (extent + 1) + 1
    """
    n = int(extent)
    uv = [
        (u, v)
        for u in range(-n, n + 1)
        for v in range(max(-n, -n - u), min(n, n - u) + 1)
    ]
    return np.asarray(uv, dtype=np.float64)


def _hex_to_xy(uv: np.ndarray) -> np.ndarray:
    """
    Axial hex (u, v) -> cartesian, unit nearest-neighbour spacing.
    """
    u, v = uv[:, 0], uv[:, 1]
    return np.column_stack([u + 0.5 * v, (np.sqrt(3.0) / 2.0) * v])


def flyvis_lattice(extent: int = 15, flip_x: bool = False, flip_y: bool = False, swap_xy: bool = False) -> 'SamplingGrid':
    """
    flyvis hex lattice as a SamplingGrid target. Handedness lives here.
    """
    xy = _hex_to_xy(flyvis_hex_uv(extent))

    return SamplingGrid.from_positions2d(
        xy,
        name=f'flyvis(extent={extent})',
        flip_x=flip_x,
        flip_y=flip_y,
        swap_xy=swap_xy,
    )


def plot_flyvis_frame(
        values: Union['SignalView', ArrayLike],
        extent: int = 15,
        hex_uv: Optional[ArrayLike] = None,
        ax: Optional['Axes'] = None,
        cmap: str = 'gray',
        edgecolor: Optional[str] = None,  # None for seamless, or a colour ('#000000', 'black', 'k', etc) for outlines
        gamma: float = 2.2
) -> 'Axes':
    """
    Scatter one flyvis frame (n_hexals,) on the hex lattice.
    Accepts the raw (H,) vector or the flyvis-shaped array (it is flattened).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    uv = flyvis_hex_uv(extent) if hex_uv is None else np.asarray(hex_uv, float)
    xy = _hex_to_xy(uv)

    # Unpack colours to stop Matplotlib from misinterpreting a VisualOutput's 'gain' channel as alpha
    if hasattr(values, 'colours') and getattr(values, 'shape', (0,))[-1] == 4:
        v = np.asarray(values.colours)
    else:
        v = np.asarray(getattr(values, 'data', values))

    if v.ndim == 1 or (v.ndim == 2 and v.shape[1] == 1):
        v = v.reshape(-1)
        is_color = False
    elif v.ndim == 2 and v.shape[1] in (3, 4):
        is_color = True
    else:
        raise ValueError(f'Unsupported shape {v.shape} for plot_flyvis_frame. Expected (T,) or (T, C)')

    if v.shape[0] != xy.shape[0]:
        raise ValueError(f'Values length ({v.shape[0]}) != hexals ({xy.shape[0]}), check extent!')

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    radius = 1 / np.sqrt(3)
    angles = np.linspace(0, 2 * np.pi, 7) + np.pi / 6
    master_hex = np.column_stack([np.cos(angles), np.sin(angles)]) * radius

    verts = xy[:, np.newaxis, :] + master_hex[np.newaxis, :, :]

    # Anti-aliasing workaround: 'face' draws a mini border of the polygon's colour over the gaps left by matplotlib
    edgecolor = str(edgecolor).lower() or 'none'
    edge_c = 'face' if edgecolor == 'none' else edgecolor

    if is_color:
        v = np.clip(v, 0.0, 1.0)
        # Apply Gamma correction
        if gamma and gamma != 1.0:
            if v.shape[1] == 4:
                v[:, :3] = np.power(v[:, :3], 1.0 / gamma)
            else:
                v = np.power(v, 1.0 / gamma)
        pc = PolyCollection(verts, edgecolors=edge_c, facecolors=v, linewidths=0.5, antialiaseds=True)
    else:
        if gamma and gamma != 1.0:
            v = np.power(np.clip(v, 0.0, None), 1.0 / gamma)
        pc = PolyCollection(verts, edgecolors=edge_c, cmap=cmap, linewidths=0.5, antialiaseds=True)
        pc.set_array(v)

    ax.add_collection(pc)

    ax.set_aspect('equal')

    ax.set_title('FlyVis BoxEye view')

    ax.set_xlim(xy[:, 0].min() - 1, xy[:, 0].max() + 1)
    ax.set_ylim(xy[:, 1].min() - 1, xy[:, 1].max() + 1)
    ax.axis('off')

    return ax


## Example use

if __name__ == '__main__':
    from rhabdoforge.types import WORLD_FORWARD
    from rhabdoforge.engine import Context, Agent, Scene, Asset
    from rhabdoforge.compound_eyes import Model
    from rhabdoforge.renderers import Renderer
    from rhabdoforge.compound_eyes.rhabdomeres import drosophila_bundle

    model = Model.from_file(
        'assets/drosophila_scaffold.npz',
        bundle=drosophila_bundle(),
        flow_direction=np.array([0.0, np.sin(np.deg2rad(10.1)), np.cos(np.deg2rad(10.1))]),
        neural_superposition=True
    )

    context = Context()

    scene = Scene()
    scene.add_instance(Asset.from_file(name='seville', file_path='assets/environments/seville_filtered.ply', radii=0.01))
    scene.add_sky('assets/textures/kloppenheim_05_4k.exr')

    renderer = Renderer(model=model, scene=scene, agent=Agent())

    for dt in context.run_headless(2):
        visual_output = renderer.step()

    ## Do the remap

    # Define a target (e.g. FlyVis)
    extent = 15
    target = flyvis_lattice(extent=extent)

    # Extract a FOV cone from the full model for instance (both eyes)
    # We query a 45-degree cone looking straight forward
    forward_fov = model.query_cone(center_direction=WORLD_FORWARD, angle=45.0, degrees=True)

    # Create the remapper from this view
    # We use k=1 and a small sigma for Gaussian blur only to smooth the overlapping L/R ommatidia
    remap = forward_fov.remapper(target, k=1, method='topology')

    # Grab the signal and remap
    # We can just pass the full signal array, the mapper will safely extract only the FOV indices
    full_signal = visual_output.peripheral_signal.colours
    signal_remapped = remap(full_signal, sigma=0.1)       # sigma should ideally be the IOA

    # Plot
    plot_flyvis_frame(signal_remapped, extent=extent)
    visual_output.plot(pathway='all')

    # Note:
    #
    # # GOOD: Stereographic projection works fine on a < 180 deg cone
    # forward_fov = model.query_cone(center_direction=[0.0, 1.0, 0.0], angle=45.0)
    # remap = forward_fov.remapper(target, k=3, method='topology')
    #
    # # BAD: A 360-degree sphere projects to infinity, causing extreme distortion/NaNs
    # remap = model.remapper(target, k=3, method='topology')


    ## Or

    # Create a low-res insect eye model to test
    low_res_model = Model.from_sphere(n=100)

    # Create a target grid from the low_res_model
    target_grid = SamplingGrid.from_model(low_res_model)

    # Create a 3D remapper from the high_res_model
    # method='3d' ensures it matches exact 3D lines of sight, no normalisation
    remapper = model.remapper(target_grid, k=1, method='3d')

    # Map the signal
    downsampled_signal = remapper(visual_output.peripheral_signal.colours)


    ## Or, example for the ContinuousResampler: Camera image -> Eye model
    #
    #
    # import cv2
    #
    # eye_sampler = ContinuousResampler.from_model(left_eye)
    # cap = cv2.VideoCapture('flight_video_360.mp4')
    #
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if not ret: break
    #
    #     ommatidia_rgb = eye_sampler.sample(frame)
