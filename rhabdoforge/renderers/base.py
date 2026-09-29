import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import TYPE_CHECKING, Optional, Union, Dict, Tuple, Sequence, Any, List
import glfw
from pathlib import Path
import random
import numpy as np
from pyglm import glm
from pytinybvh import BVH

from rhabdoforge.types import (
    EyeOutput, OmmatidiaProjection, OverlayColormap, DisplayMode, RandomnessMode, SamplingMode, to_enum, OMM_STATIC_DTYPE,
    OMM_DYNAMIC_DTYPE, RHAB_STATIC_DTYPE, RHAB_DYNAMIC_DTYPE
)
from rhabdoforge.LUTs import airy_sensitivity_lut, waveguide_sensitivity_lut
from rhabdoforge.engine.meshes import CONE_VERTICES, SPHERE_VERTICES
from rhabdoforge.engine.resources import (
    ShaderProgram, GPUResourceManager, BufferRegistry, UniformRegistry, TextureRegistry, HDRRenderTarget,
    TextureViewer, StaticRenderTarget
)
from rhabdoforge.engine.materials_utils import constant_sh
from rhabdoforge.renderers.baking import SceneBaker, TEX_TIERS
from rhabdoforge.renderers.helpers import VisualOutput, HistoryRecorder

if TYPE_CHECKING:
    from rhabdoforge.engine.scene import Scene
    from rhabdoforge.engine.agent import Agent, OrbitCamera
    from rhabdoforge.engine.context import Context
    from rhabdoforge.compound_eyes import Model, Eye
    from rhabdoforge.compound_eyes.views import BaseView


WORKGROUPS_DYNAMICS = 64


def query_available_VRAM() -> int:
    """Queries available VRAM in MB."""

    extensions = glGetString(GL_EXTENSIONS).decode('utf-8').split()
    avail = 0

    # NVIDIA
    if 'GL_NVX_gpu_memory_info' in extensions:
        from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
        avail = glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) // 1024

    # AMD
    elif 'GL_ATI_meminfo' in extensions:
        # GL_VBO_FREE_MEMORY_ATI returns [total_free, largest_free_block, total_aux_free, largest_aux_free]
        GL_VBO_FREE_MEMORY_ATI = 0x87FB
        mem_info = (GLint * 4)()
        glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, mem_info)
        avail = mem_info[0] // 1024

    if not avail:
        print('WARNING: Could not query available VRAM. Assuming enough memory is available.')

    return avail


##


class Renderer:
    """
    Compound-eye rendering (single-bounce ray tracing by default, multi-bounce path-tracing optional).
    """

    def __init__(self,
                 model: 'Model',
                 scene: 'Scene',
                 agent: 'Agent',
                 time_dithering: bool = True,
                 nb_samples: int = 256,
                 randomness_mode: Union[int, str, RandomnessMode] = RandomnessMode.Pseudo,
                 sampling_mode: Union[int, str, SamplingMode] = SamplingMode.Gaussian,
                 panoramic_resolution: Optional[Tuple[int, int]] = (1024, 512),
                 batch_size: int = 1,
                 track_history: bool = False,
                 history_path: Optional[Union[str, Path]] = None,
                 max_bounces: int = 0,
                 enable_microsaccades: bool = False,
                 enable_direct: bool = True,
                 enable_shadows: bool = True,
                 enable_ambient: bool = True,
                 capture_raw_depth: bool = False,
                 resource_manager: Optional['GPUResourceManager'] = None,
                 context: Optional['Context'] = None
                 ):

        from rhabdoforge.engine.context import get_context
        self._context = context or get_context()

        # Scene + baker (VRAM estimation depends on it)
        self._scene: 'Scene' = scene
        self._resource_manager: 'GPUResourceManager' = resource_manager or GPUResourceManager()
        self._baker: 'SceneBaker' = SceneBaker(scene, self._resource_manager)

        # Compound eyes model and agent
        self._model: 'Model' = model
        self.agent: 'Agent' = agent

        self._context.renderer = self

        # Track view matrices for parciminious updates
        self._last_view_matrix = None
        self._last_persp_view_matrix = None

        self._latest_output: Optional['VisualOutput'] = None    # only for dashboard, etc

        # Main sampling parameters
        self._max_bounces: int = max_bounces
        self._samples_per_rhab: int = 1     # number of rays per rhabdomere
        self._samples_per_px: int = 1       # number of rays per pixel (third person visualisation only)
        self._noise_threshold = 0.05
        self._use_hybrid_sampling = False
        self._randomness_mode = to_enum(randomness_mode, RandomnessMode)
        self._sampling_mode = to_enum(sampling_mode, SamplingMode)
        self._capture_raw_depth = capture_raw_depth

        # Render surfaces and related things
        self._bg_col_linear = tuple(c ** 2.2 for c in self.scene.background_color)  # TODO: what if already linear
        self._screen_surface: Optional['TextureViewer'] = None      # onscreen, visible surface
        self._panoramic_resolution = panoramic_resolution

        # Offscreen surfaces
        self._hdr_surface: 'HDRRenderTarget' = HDRRenderTarget()    # Main HDR render target
        self._snapshot_target: 'StaticRenderTarget' = StaticRenderTarget(srgb=True)     # Render target for screenshots

        self._tonemap_shader = ShaderProgram(vert_path='fullscreen.vert', frag_path='tonemap.frag')
        self._tonemap_vao = glGenVertexArrays(1)
        self._exposure = 1.0

        # Slots for lazy resource handles
        self._current_defines: Dict[str, Any] = {}
        self._projection_shaders: Dict[str, 'ShaderProgram'] = {}
        self._eyemesh_shaders: Dict[Tuple[str, bool], 'ShaderProgram'] = {}
        self._eyemesh_vaos: Dict[str, Tuple[int, int]] = {}  # shape_name -> (vao_id, vertex_count)

        # Get workable memory sizes
        self._max_ssbo_bytes = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
        self._batch_size, self._samples_per_rhab = self._safe_samples_lim(
            batch_size, nb_samples, prioritize_batch=True
        )
        self._track_history: bool = track_history
        self._history_path = Path(history_path) if history_path is not None else None
        self._history: Optional['HistoryRecorder'] = None

        # Initialise the registries
        self._eye_uniforms = UniformRegistry()
        self._lights_uniforms = UniformRegistry()
        self._scene_uniforms = UniformRegistry()

        self.eye_buffers: 'BufferRegistry' = BufferRegistry(self._resource_manager)
        self._projection_textures: 'TextureRegistry' = TextureRegistry(self._resource_manager)

        # Store local attributes to sync with uniforms

        # Global lighting controls
        self._enable_direct = enable_direct
        self._enable_shadows = enable_shadows
        self._enable_ambient = enable_ambient
        self._ambient_intensity = 1.0
        self._sky_intensity = 1.0

        # States flags and other things
        self._time_dithering: bool = time_dithering
        self._overlay_enabled: bool = False
        self.runs_interactive: bool = False

        self._microsaccades_enabled: bool = None
        self.microsaccades_enabled = enable_microsaccades
        self._override_pupil_drive: Optional[float] = None
        self._override_saccade_drive: Optional[float] = None

        self.false_colours: bool = False            # TODO: expose these (and generate UV-encoded assets for demo)
        self.uv_encoded_textures: bool = False

        # Time keeping
        self._dither_counter: int = 0   # only advanced when time dithering is on
        self._frame_index: int = 0      # advanced at each new rendered frame

        # Eye mesh visualisation
        self._omm_length_factor = 1.0
        self._eyes_exaggeration = 1.0           # 1.0 = render the eye at true physical size
        self._saccade_exaggeration = 1e-6       # TODO: why
        self._rf_exaggeration = 0.5             # 0.5 = RF blobs tile edge-to-edge (tiled mode)

        # Other visualisation state and parameters stuff
        self._projection_mode = OmmatidiaProjection.Position
        self._output_mode: 'EyeOutput' = EyeOutput.Cartridge
        self._tiled_mode = True
        self._lum_ref = 1.0             # target operating-point luminance (scene-dependant)
        self._noise_threshold = 0.05
        self._trigger_delay = 0.008     # reflex arc onset latency (s), Juusola 2017 Appendix 8
        self._selected_omm_indices = np.full(10, -1, dtype=np.int32)

        # Overlay parameters
        self._overlay_colormap: 'OverlayColormap' = OverlayColormap.Thermal
        self._overlay_range: Tuple[float, float] = (0.0, 1.0)
        self._overlay_current_peak: float = 0.0
        self._overlay_compression: float = 1.0         # power exponent for range compression (1.0 = linear, 0.5 = sqrt, etc)
        self._overlay_autorange_perc: int = 98       # percentile to reject outliers

        # TODO: Add multiple selectable overlay modes: luminance, adaptation state, etc, and custom (the set_data one)

        using_sky = self.scene.sky is not None
        self._scene_uniforms.update(
            nb_tlas_nodes=len(self._baker.cpu_tlas_nodes),
            background_color=self.scene.background_color,
            max_bounces=self._max_bounces,

            # Sky params
            use_sky=using_sky,
            sky_texture=self._baker.scene_textures['sky_texture'].unit if using_sky else 0,
            sh_irradiance_coeffs=self.scene.sky.sh_coeffs if using_sky else constant_sh(self.scene.background_color),


            # One material texture array per size tier
            **{f'scene_textures_{i}': self._baker.tex_arrays[tier].unit if tier in self._baker.tex_arrays else 0
               for i, tier in enumerate(TEX_TIERS)}
        )

        self._lights_uniforms.update(
            enable_ambient=self._enable_ambient,
            enable_direct=self._enable_direct,
            enable_shadows=self._enable_shadows,
            sky_intensity=self._sky_intensity,
            ambient_intensity=self._ambient_intensity,
            directional_lights_count=self._baker._nb_dir_lights,
            point_lights_count=self._baker._nb_point_lights,
            area_lights_count=self._baker._nb_area_lights
        )

        self._init_model_resources()

        self._update_selected_ommatidia()

    def __repr__(self) -> str:
        loop_mode = 'Open-loop (batched)' if self._batch_size > 1 else 'Closed-loop / Interactive'
        return (f"<{self.__class__.__name__} | Mode: {loop_mode} | "
                f"Batch size: {self._batch_size} | "
                f"{self.nb_samples} samples/rhabdomere>")

    def _init_model_resources(self) -> None:

        # Allocate GPU buffers
        rays_elements = self._model.N if self._model.bundle.fused_rhabdoms else self._model.size

        # Ping-pong PBOs
        self.eye_buffers.allocate('pbo_0',
                                  dtype=np.uint8,
                                  count=self._model.size * 16,
                                  target=GL_PIXEL_PACK_BUFFER,
                                  usage=GL_STREAM_READ)
        self.eye_buffers.allocate('pbo_1',
                                  dtype=np.uint8,
                                  count=self._model.size * 16,
                                  target=GL_PIXEL_PACK_BUFFER,
                                  usage=GL_STREAM_READ)

        # SSBOs
        self.eye_buffers.allocate('rhab_static',
                                  dtype=RHAB_STATIC_DTYPE,
                                  count=self._model.size,
                                  data=self._model.buffer.rhabdomere_static,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('omm_static',
                                  dtype=OMM_STATIC_DTYPE,
                                  count=self._model.N,
                                  data=self._model.buffer.ommatidia_static,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('rhab_dynamic',
                                  dtype=RHAB_DYNAMIC_DTYPE,
                                  count=self._model.size,
                                  data=self._model.buffer.rhabdomere_dynamic,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('omm_dynamic',
                                  dtype=OMM_DYNAMIC_DTYPE,
                                  count=self._model.N,
                                  data=self._model.buffer.ommatidia_dynamic,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('ema_state',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers['ema_state'].reset()

        # Wave optics SSBO if needed
        waveguide_lut = self._get_waveguide_lut()

        self.eye_buffers.allocate('waveguide_lut',
                                  dtype=np.float32,
                                  count=waveguide_lut.size,
                                  data=waveguide_lut,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('rays_intermediate',
                                  dtype=np.dtype([
                                      ('color_weight', np.float32, (4,)),
                                      ('depth_data', np.float32, (4,)),
                                  ]),
                                  count=rays_elements * self._samples_per_rhab,
                                  usage=GL_DYNAMIC_DRAW)

        # Async SSBO
        self.eye_buffers.allocate('colors',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size * self._batch_size,
                                  usage=GL_DYNAMIC_DRAW,
                                  supports_async=True,
                                  _async_reader=self._readback_async)
        self.eye_buffers.allocate('depth',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size * self._batch_size,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('depth_stats_b',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size * self._batch_size,
                                  usage=GL_DYNAMIC_DRAW)

        # All buffers allocated (baker ones + eye_buffers ones): Compile the main shaders
        self._current_defines = self._collect_defines()
        self.dispatch_shader = ShaderProgram(comp_path='shaders/dispatch.comp', defines=self._current_defines)
        self.reduction_shader = ShaderProgram(comp_path='shaders/reduction.comp', defines=self._current_defines)
        self.dynamics_shader = ShaderProgram(comp_path='shaders/dynamics.comp', defines=self._current_defines)

        # Tracking of PBOs state
        self._pbo_index = 0
        self._fences = [0, 0]
        self._colours_cpu_buffer = np.zeros((self._model.size, 4), dtype=np.float32)

        # Upload all default uniforms values

        self._eye_uniforms.update(
            aspect_ratio=1.0,

            # Rhabdomeres and ommatidia (constants during runtime)
            nb_ommatidia=self._model.N,
            nb_rhabdomeres=self._model.size,
            rhab_per_omm=self._model.R,
            bundle_centre_idx=self._model.bundle.center_index,
            fused_rhabdoms=int(self._model.bundle.fused_rhabdoms),

            # Various visualisation parameters
            visualisation_eye_surface_albedo=1.0,
            **self._update_visualisation_scales(),

            # States
            output_mode=self._output_mode,
            tiled_mode=self._tiled_mode,
            projection_mode=self._projection_mode,
            false_colors=self.false_colours and not self.uv_encoded_textures,
            uv_encoding=self.uv_encoded_textures,

            # Sampling modes
            nb_samples=self._samples_per_rhab,
            pixel_samples=self._samples_per_px,
            use_hybrid_sampling=self._use_hybrid_sampling,
            sampling_mode=self._sampling_mode,  # 0 = Gaussian, 1 = Airy, 2 = Waveguide
            randomness_mode=self._randomness_mode,
            airy_lut=airy_sensitivity_lut(),

            # Visualisation defaults
            overlay_fallback=True,
            overlay_data_min=self._overlay_range[0],
            overlay_data_max=self._overlay_range[1],
            overlay_colormap=int(self._overlay_colormap),
            overlay_compression=self._overlay_compression,

            # Time tracking
            dt=0.0,
            frame_offset=self._frame_index % self._batch_size,
            dither_counter=self._dither_counter,

            # Rhabdomere dynamics
            noise_threshold=self._noise_threshold,
            enable_actuation=self._microsaccades_enabled,
            photon_concentration_factor=0.0,  # TODO: document this better
            lum_ref=self._lum_ref,
            clip_ratio=float(self._model.bundle.clip_ratio),
            force_pupil_drive=self._override_pupil_drive is not None,
            pupil_drive=0.0,
            force_saccade_drive=self._override_saccade_drive is not None,
            saccade_drive=0.0,
            trigger_delay=self._trigger_delay,
            aperture_follow=float(self._model.bundle.aperture_follow),
        )

    def _free_model_resources(self) -> None:
        """
        Tear down everything sized by or derived from the model
        """

        # Compute shaders: dispatch + panoramic/perspective projection
        self._invalidate_shaders()

        # Reduction / dynamics compute shaders
        for shader in (self.reduction_shader, self.dynamics_shader):
            if shader:
                shader.free()
        self.reduction_shader = None
        self.dynamics_shader = None

        # PBO ping-pong sync fences
        for f in self._fences:
            if f:
                try:
                    glDeleteSync(f)
                except Exception:
                    pass
        self._fences = [0, 0]
        self._pbo_index = 0

        # Eye-mesh VAOs reference VBOs that live inside eye_buffers so they must go before the registry is freed
        for vao, _ in self._eyemesh_vaos.values():
            glDeleteVertexArrays(1, [vao])
        self._eyemesh_vaos.clear()

        # Eye mesh shaders rebuild lazily
        for shader in self._eyemesh_shaders.values():
            shader.free()
        self._eyemesh_shaders.clear()

        # All per-ommatidium / per-rhabdomere SSBOs + PBOs and the lazy overlay / eye mesh VBOs that share the registry
        self.eye_buffers.free()

    def _update_model_uniforms(self) -> None:
        """Upload model-derived uniforms."""

        self._eye_uniforms.update(
            nb_ommatidia=self._model.N,
            nb_rhabdomeres=self._model.size,
            rhab_per_omm=self._model.R,
            bundle_centre_idx=self._model.bundle.center_index,
            fused_rhabdoms=int(self._model.bundle.fused_rhabdoms),
            clip_ratio=float(self._model.bundle.clip_ratio),
            aperture_follow=float(self._model.bundle.aperture_follow),
            **self._update_visualisation_scales()
        )

    def _update_selected_ommatidia(self) -> None:
        sel_omm_indices = self._selected_omm_indices.copy()

        if self._output_mode == EyeOutput.Raw:
            for i in range(10):
                if sel_omm_indices[i] != -1:
                    sel_omm_indices[i] *= self._model.R
                    sel_omm_indices[i] += self._model.bundle.center_index

        self._eye_uniforms.update(selected_ommatidia=sel_omm_indices)

    def _invalidate_shaders(self) -> None:
        """Invalidates all shaders that need be when defines change."""

        self._current_defines = self._collect_defines()

        self.dispatch_shader.free()
        self.dispatch_shader = ShaderProgram(comp_path='shaders/dispatch.comp', defines=self._current_defines)

        self.reduction_shader.free()
        self.reduction_shader = ShaderProgram(comp_path='shaders/reduction.comp', defines=self._current_defines)

        for s in self._projection_shaders.values():
            s.free()

        self._projection_shaders.clear()

    # Internal getters for lazy loaded resources

    def _get_eyemesh_vao(self, shape: str) -> Tuple[int, int]:
        """Lazy-loads and returns (vao_id, vertex_count) for the requested shape."""

        if shape not in self._eyemesh_vaos:
            if shape == 'cone':
                v_data = CONE_VERTICES
                vbo_name = 'cones_vbo'
            elif shape == 'hemisphere':
                v_data = SPHERE_VERTICES
                vbo_name = 'hemisph_vbo'
            else:
                raise ValueError(f"Unknown shape: {shape}")

            vertex_count = len(v_data) // 3

            self.eye_buffers.allocate(vbo_name,
                                      dtype=np.float32,
                                      count=len(v_data),
                                      target=GL_ARRAY_BUFFER,
                                      data=v_data)

            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            with self.eye_buffers[vbo_name].bind():
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
            glBindVertexArray(0)

            self._eyemesh_vaos[shape] = (vao, vertex_count)

        return self._eyemesh_vaos[shape]

    def _get_eyemesh_shader(self, view_type: str, overlay: bool) -> 'ShaderProgram':
        """Lazy-loads first/third person shaders with or without overlay."""

        key = (view_type, overlay)

        if key not in self._eyemesh_shaders:
            prefix = 'subjective' if view_type == 'subjective' else 'external'
            defines = self.eye_buffers.shader_defines.copy()

            if overlay:
                defines['OVERLAY_MODE'] = 1

            self._eyemesh_shaders[key] = ShaderProgram(
                vert_path=f'{prefix}.vert',
                frag_path=f'{prefix}.frag',
                defines=defines
            )

        return self._eyemesh_shaders[key]

    def _get_projection_shader(self, proj_name: str) -> 'ShaderProgram':

        if self._collect_defines() != self._current_defines:
            self._invalidate_shaders()

        if proj_name not in self._projection_shaders:
            if proj_name == 'panoramic':
                shader_path = 'shaders/panoramic.comp'
            else:  # view_name == 'perspective'
                shader_path = 'shaders/perspective.comp'


            defines = {**self._current_defines, 'USE_LOD': 1} # USE_LOD enables LOD estimate. Voluntarily not used for ommatidia dispatch shader
            self._projection_shaders[proj_name] = ShaderProgram(comp_path=shader_path, defines=defines)

        return self._projection_shaders[proj_name]

    def _get_projection_texture(self, proj_name: str) -> Tuple[int, Tuple[int, int]]:

        if proj_name == 'panoramic':
            target_res = self._panoramic_resolution

        else:  # perspective
            viewport = glGetIntegerv(GL_VIEWPORT)
            target_res = (viewport[2], viewport[3])

        if proj_name not in self._projection_textures:
            self._projection_textures.allocate_2d(proj_name, target_res[0], target_res[1], dtype=float)

        return self._projection_textures[proj_name].handle, target_res

    def _collect_defines(self) -> Dict[str, Any]:
        """
        Determines which #defines to inject based on current baker light counts.
        """

        defines = {}

        defines.update(self.eye_buffers.shader_defines)
        defines.update(self._baker.bvh_buffers.shader_defines)
        defines.update(self._baker.light_buffers.shader_defines)

        for count, name in [
            (self._baker._nb_dir_lights, 'DIRECTIONAL'),
            (self._baker._nb_point_lights, 'POINT'),
            (self._baker._nb_area_lights, 'AREA')
        ]:
            if count >= 1:
                defines[f'HAS_{name}_LIGHT'] = 1
            if count > 1:
                defines[f'MULTI_{name}'] = 1

        if self._max_bounces > 0:
            defines['PATH_TRACING'] = 1

        defines['SAMPLE_GROUP_SIZE'] = self._sample_group_size()

        return defines

    def _sample_group_size(self) -> int:
        """
        Compile-time workgroup size for dispatch.comp/reduction.comp: smallest power of two
        >= nb_samples, so a rhabdomere's whole sample set fills exactly 1 workgroup.
        """
        n = max(1, self._samples_per_rhab)
        size = 1 << (n - 1).bit_length()
        return max(32, min(size, 1024))

    def _update_visualisation_scales(self) -> Dict[str, float]:
        """
        Derive eye mesh visualisation dimensions from the model's geometry.
        """

        omm = self._model.ommatidia

        ampl = float(np.mean(np.abs(omm.lateral_amplitude)))
        focal = float(np.mean(omm.focal_length))
        saccade_rad = ampl / focal if (ampl > 0 and focal > 0) else 0.0

        saccade_scale = (self._saccade_exaggeration / saccade_rad) if saccade_rad > 0 else 0.0

        return dict(
            visualisation_omm_length=self._omm_length_factor,
            visualisation_eyes_scale=self._eyes_exaggeration,
            visualisation_saccade_scale=saccade_scale,
            visualisation_rf_scale=self._rf_exaggeration,
        )

    # Various internal helpers

    def _safe_samples_lim(self, batch_size: int, nb_samples: int, prioritize_batch: bool = True) -> Tuple[int, int]:
        """
        Calculates a safe batch_size and nb_samples based on Hardware SSBO limits and available VRAM.
        """

        # Hardware limit: rays_intermediate SSBO block size
        rays_elements = self._model.N if self._model.bundle.fused_rhabdoms else self._model.size
        max_samples_hw = self._max_ssbo_bytes // (rays_elements * 16)
        safe_samples = max(1, min(nb_samples, max_samples_hw))

        if safe_samples < nb_samples:
            print(f'Warning: Clamped nb_samples to {safe_samples} (Hardware SSBO limit).')

        # VRAM limit
        avail_mb = query_available_VRAM() * 0.9  # 90 % safety margin
        if not avail_mb:  # if query failed assume we're fine
            return batch_size, safe_samples

        # Baker buffers on GPU fixed costs
        fixed_usage_bytes = sum(buf.nbytes for buf in (
            self._baker.cpu_verts, self._baker.cpu_idx, self._baker.cpu_pts,
            self._baker.cpu_blas_nodes, self._baker.cpu_tlas_nodes, self._baker.cpu_tlas_indices,
            self._baker.cpu_blas_indices, self._baker.gpu_inst_info
        ) if buf is not None)

        # PBO costs (2x model_size * 16)
        fixed_usage_bytes += self._model.size * 16 * 2
        fixed_mb = fixed_usage_bytes / (1024 * 1024)
        room_for_dynamic_mb = avail_mb - fixed_mb

        # Dynamic cost per unit (1 unit = model_size * 16 bytes)
        colors_unit_mb = (self._model.size * 16) / (1024 * 1024)
        rays_unit_mb = (rays_elements * 16) / (1024 * 1024)

        def calc_needed():
            return (batch_size * colors_unit_mb) + (safe_samples * rays_unit_mb)

        # If exceeding VRAM, shrink the one that isn't prioritised
        if calc_needed() > room_for_dynamic_mb:

            if prioritize_batch:
                # Keep batch_size, shrink samples
                max_samples_vram = int((room_for_dynamic_mb - (batch_size * colors_unit_mb)) / rays_unit_mb)
                safe_samples = max(1, min(safe_samples, max_samples_vram))

                # Still too much, must still shrink batch_size
                if calc_needed() > room_for_dynamic_mb:
                    batch_size = max(1, int((room_for_dynamic_mb - (1 * colors_unit_mb)) / colors_unit_mb))
            else:
                # Keep samples, shrink batch_size
                max_batch_vram = int((room_for_dynamic_mb - (safe_samples * rays_unit_mb)) / colors_unit_mb)
                batch_size = max(1, min(batch_size, max_batch_vram))

                # still too much, must still shrink samples
                if calc_needed() > room_for_dynamic_mb:
                    safe_samples = max(1, int((room_for_dynamic_mb - (1 * rays_unit_mb)) / rays_unit_mb))

            print(f'Warning: VRAM limit reached. Config adjusted to: Batch={batch_size}, Samples={safe_samples}')

        return batch_size, safe_samples

    # Main internal rendering calls: Dispatch -> Reduction -> Dynamics

    def _dispatch(self) -> None:

        self._baker.update()

        e = self.eye_buffers
        b = self._baker.bvh_buffers
        l = self._baker.light_buffers

        with self.dispatch_shader as shader:

            with b.grouped_bind(), l.grouped_bind(), e.grouped_bind(['rays_intermediate', 'rhab_static', 'omm_static', 'rhab_dynamic', 'waveguide_lut']):

                with self._baker.scene_textures.bind_all():

                    current_view = self.agent.view
                    if current_view != self._last_view_matrix:
                        self._eye_uniforms.update(cam_to_world=glm.inverse(current_view))
                        self._last_view_matrix = current_view

                    self._eye_uniforms.apply(shader)
                    self._scene_uniforms.apply(shader)
                    self._lights_uniforms.apply(shader)

                    rays_elements = self._model.N if self._model.bundle.fused_rhabdoms else self._model.size
                    glDispatchCompute(rays_elements, 1, 1)  # 1 workgroup per rhabdomere's sample set

                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _reduction(self) -> None:

        with self.reduction_shader as shader:
            with self.eye_buffers.grouped_bind(
                    ['rays_intermediate', 'rhab_static', 'colors', 'depth',
                     'depth_stats_b', 'ema_state', 'rhab_dynamic']):
                self._eye_uniforms.apply(shader)

                rays_elements = self._model.N if self._model.bundle.fused_rhabdoms else self._model.size
                glDispatchCompute(rays_elements, 1, 1)

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _dynamics(self) -> None:

        with self.dynamics_shader as shader:

            with self.eye_buffers.grouped_bind(['rhab_static', 'omm_static', 'colors', 'ema_state', 'rhab_dynamic', 'omm_dynamic']):

                self._eye_uniforms.apply(shader)

                work_groups = (self._model.N + WORKGROUPS_DYNAMICS - 1) // WORKGROUPS_DYNAMICS
                glDispatchCompute(work_groups, 1, 1)

                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _readback_async(self) -> np.ndarray:
        """
        Non-blocking colour readback (via ping-pong PBO ring).
        Returns the *previous* frame colours (and zeros on the first frame).
        """

        bytes_to_read = self._model.size * 16

        # Copy colour SSBO into current PBO
        with (self.eye_buffers['colors'].bind(mode_override=GL_COPY_READ_BUFFER),
              self.eye_buffers[f'pbo_{self._pbo_index}'].bind(mode_override=GL_COPY_WRITE_BUFFER)):

            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes_to_read)

        # Fence to know when the copy is done
        if self._fences[self._pbo_index]:
            glDeleteSync(self._fences[self._pbo_index])
        self._fences[self._pbo_index] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

        # Read from the other PBO (previous frame data)
        next_pbo_index = (self._pbo_index + 1) % 2
        fence = self._fences[next_pbo_index]

        if fence:

            with self.eye_buffers[f'pbo_{next_pbo_index}'].bind(mode_override=GL_PIXEL_PACK_BUFFER):
                ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes_to_read, GL_MAP_READ_BIT)

                if ptr:
                    ctypes.memmove(self._colours_cpu_buffer.ctypes.data, ptr, bytes_to_read)
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                    out_array = self._colours_cpu_buffer.copy()
                else:
                    print('Warning: Failed to map PBO. Context lost?')
                    out_array = np.zeros_like(self._colours_cpu_buffer)
        else:
            out_array = np.zeros_like(self._colours_cpu_buffer)

        self._pbo_index = next_pbo_index
        return out_array

    def _read_outputs_sync(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Read colour and depth for the same completed frame.

        The legacy colour path is ping-pong asynchronous and therefore returns
        the previous frame. Depth must be paired with the same visual frame,
        so the depth-enabled path uses a synchronous read for now.
        """
        glFinish()
        with self.eye_buffers['colors'].bind():
            colour_bytes = glGetBufferSubData(
                GL_SHADER_STORAGE_BUFFER, 0, self._model.size * 16
            )
        depth_values = self.eye_buffers['depth'].read(count=self._model.size)
        count_values = self.eye_buffers['depth_stats_b'].read(count=self._model.size)
        colours = np.frombuffer(colour_bytes, dtype=np.float32).reshape(self._model.size, 4).copy()
        count = count_values[:, 0].copy()
        valid = count > 0.0
        total = depth_values[:, 0]
        mean = np.full(count.shape, np.nan, dtype=np.float32)
        mean[valid] = total[valid] / count[valid]
        variance = np.full(count.shape, np.nan, dtype=np.float32)
        variance[valid] = np.maximum(
            depth_values[valid, 1] / count[valid] - mean[valid] ** 2, 0.0
        )
        return colours, {
            'mean': mean,
            'min': np.where(valid, depth_values[:, 2], np.nan),
            'max': np.where(valid, depth_values[:, 3], np.nan),
            'std': np.sqrt(variance),
            'valid_count': count.astype(np.int32),
        }

    def _read_raw_depth_sync(self) -> np.ndarray:
        """Read per-ray primary depths for an explicitly diagnostic render."""
        if not self._capture_raw_depth:
            raise RuntimeError('raw depth capture is disabled')
        rays = self.eye_buffers['rays_intermediate'].read()
        rays_elements = self._model.N if self._model.bundle.fused_rhabdoms else self._model.size
        return rays['depth_data'][:, 0].reshape(rays_elements, self._samples_per_rhab)

    def _tonemap_pass(self) -> None:

        shader = self._tonemap_shader

        with shader:
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)

            glActiveTexture(GL_TEXTURE0)

            glBindTexture(GL_TEXTURE_2D, self._hdr_surface.color)
            glUniform1i(shader.get_loc('hdr_scene'), 0)
            glUniform1f(shader.get_loc('exposure'), float(self.exposure))
            glBindVertexArray(self._tonemap_vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)
            glBindVertexArray(0)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)

    # Internal render calls for optional visualisation modes

    def _render_subjective_view(self) -> None:
        """First-person compound-eye view (colours or scalar overlay)."""

        shader = self._get_eyemesh_shader('subjective', self.overlay_enabled)

        with shader:

            glEnable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            glDepthFunc(GL_LEQUAL)

            # Need the first person projection to conform to viewport aspect
            viewport = glGetIntegerv(GL_VIEWPORT)
            self._eye_uniforms.update(aspect_ratio=viewport[2] / viewport[3] if viewport[3] > 0 else 1.0)

            self._eye_uniforms.apply(shader)

            to_bind = ['rhab_static', 'omm_static', 'rhab_dynamic', 'omm_dynamic', 'colors']
            if self.overlay_enabled:
                to_bind.append('overlay')

            with self.eye_buffers.grouped_bind(to_bind):
                nb_units = self._model.size if self.output_mode == EyeOutput.Raw else self._model.N

                vao, vertex_count = self._get_eyemesh_vao('cone')

                glBindVertexArray(vao)
                glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count, nb_units)
                glBindVertexArray(0)

            glDisable(GL_DEPTH_TEST)

    def _render_external_view(self, observer_camera) -> None:
        """Third-person eye model (colours or scalar overlay)."""

        shader = self._get_eyemesh_shader('external', self.overlay_enabled)

        with shader:

            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_CULL_FACE)

            self._eye_uniforms.update(
                view=observer_camera.view,
                projection=observer_camera.projection,
                eye_to_world=glm.inverse(self.agent.view)
            )

            self._eye_uniforms.apply(shader)

            to_bind = ['rhab_static', 'omm_static', 'rhab_dynamic', 'omm_dynamic', 'colors']
            if self.overlay_enabled:
                to_bind.append('overlay')

            with self.eye_buffers.grouped_bind(to_bind):
                nb_units = self._model.size if self.output_mode == EyeOutput.Raw else self._model.N

                shape = 'hemisphere' if self.projection_mode == OmmatidiaProjection.Position else 'cone'
                vao, vertex_count = self._get_eyemesh_vao(shape)

                glBindVertexArray(vao)
                glDrawArraysInstanced(GL_TRIANGLES, 0, vertex_count, nb_units)
                glBindVertexArray(0)

            glEnable(GL_CULL_FACE)
            glDisable(GL_BLEND)
            glDisable(GL_DEPTH_TEST)

    def _raytrace_thirdperson(self, view_name: str, pov: Union['Agent', 'OrbitCamera']) -> None:
        """Shared dispatch for panoramic / perspective ray-traced views."""

        tex_id, res = self._get_projection_texture(view_name)

        b = self._baker.bvh_buffers
        l = self._baker.light_buffers

        with self._get_projection_shader(view_name) as shader:
            glBindImageTexture(0, tex_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

            with b.grouped_bind(), l.grouped_bind():
                with self._baker.scene_textures.bind_all():

                    current_view = pov.view
                    if current_view != self._last_view_matrix:
                        self._eye_uniforms.update(cam_to_world=glm.inverse(current_view))
                        self._last_view_matrix = current_view

                    if view_name == 'perspective':
                        curr_persp = self.agent.projection
                        if curr_persp != self._last_persp_view_matrix:
                            self._eye_uniforms.update(inv_projection=glm.inverse(curr_persp))
                            self._last_persp_view_matrix =curr_persp
                        pixel_angular_size = glm.radians(self.agent.fov) / res[1]
                    else:
                        pixel_angular_size = np.pi / res[1]  # Pano: 180 deg vertical

                    self._eye_uniforms.update(pixel_angular_size=pixel_angular_size)

                    self._eye_uniforms.apply(shader)
                    self._scene_uniforms.apply(shader)
                    self._lights_uniforms.apply(shader)

                    glDispatchCompute((res[0] + 15) // 16, (res[1] + 15) // 16, 1)
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def flush(self) -> Optional['VisualOutput']:
        """
        Blocks until all queued frames on the GPU are rendered, downloads the data, and resets the counter.
        """
        if self._frame_index == 0:
            return None

        # Block until all rendering commands are complete
        t0 = glfw.get_time()
        glFinish()
        stall_time = glfw.get_time() - t0

        # Account for GPU stall time in the context clock
        if self._context is not None:
            self._context._total_wall_time += stall_time
            self._context._last_wall_time = glfw.get_time()

        frames_to_read = min(int(self._frame_index), self._batch_size)

        # Download the block
        with self.eye_buffers['colors'].bind():
            data_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self._model.size * 16 * frames_to_read)
            data_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(frames_to_read, self._model.size, 4)

        self._frame_index = 0
        return VisualOutput(data=data_np, model=self._model)

    def _record_history(self, out: 'VisualOutput') -> None:
        """Appends a frame or batch chunk to the disk-backed history recorder (created lazily)."""
        if self._history is None:
            self._history = HistoryRecorder(
                frame_shape=out.data.shape[-2:],
                dtype=out.data.dtype,
                path=self._history_path
            )
        self._history.extend(out.data)

    def sync_cpu(self, force_all=False) -> None:
        """
        CPU data sync (contiguous block synchronisation).
        Checks the high-level flag first, then walks the mask for surgical uploads.
        """
        buf = self._model.buffer

        # Ommatidia
        if buf.ommatidia_stale or force_all:
            if force_all:
                self.eye_buffers['omm_static'].write(buf.ommatidia_static)
                self.eye_buffers['omm_dynamic'].write(buf.ommatidia_dynamic)
            else:
                dirty_idx = np.where(buf.ommatidia_stale_mask)[0]
                if dirty_idx.size > 0:
                    jumps = np.where(np.diff(dirty_idx) != 1)[0] + 1
                    for block in np.split(dirty_idx, jumps):
                        start, nb = block[0], block.size
                        self.eye_buffers['omm_static'].write(buf.ommatidia_static[start:start + nb], start=start)
                        self.eye_buffers['omm_dynamic'].write(buf.ommatidia_dynamic[start:start + nb], start=start)

            buf.ommatidia_stale = False
            buf.ommatidia_stale_mask.fill(False)

        # Rhabdomeres
        if buf.rhabdomeres_stale or force_all:
            if force_all:
                self.eye_buffers['rhab_static'].write(buf.rhabdomere_static)
                self.eye_buffers['rhab_dynamic'].write(buf.rhabdomere_dynamic)
            else:
                dirty_idx = np.where(buf.rhabdomeres_stale_mask)[0]
                if dirty_idx.size > 0:
                    jumps = np.where(np.diff(dirty_idx) != 1)[0] + 1
                    for block in np.split(dirty_idx, jumps):
                        start, nb = block[0], block.size
                        self.eye_buffers['rhab_static'].write(buf.rhabdomere_static[start:start + nb], start=start)
                        self.eye_buffers['rhab_dynamic'].write(buf.rhabdomere_dynamic[start:start + nb], start=start)

            buf.rhabdomeres_stale = False
            buf.rhabdomeres_stale_mask.fill(False)

    # Main public methods

    def draw(self, view_mode: Union[str, 'DisplayMode'], point_of_view: Union['Agent', 'OrbitCamera'],
             target_fbo: int = 0, override_size: Optional[Tuple[int, int]] = None, bg_alpha: float = 1.0):

        w, h = override_size if override_size else self.context.viewport_size
        self._hdr_surface.bind(w, h)

        glClearColor(*self._bg_col_linear, bg_alpha)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if view_mode == DisplayMode.Compound:
            self._render_subjective_view()

        elif view_mode in (DisplayMode.Panoramic, DisplayMode.Perspective, DisplayMode.Third_person):
            n = 'panoramic' if view_mode == DisplayMode.Panoramic else 'perspective'

            self._baker.update()
            self._raytrace_thirdperson(n, point_of_view)

            tex_id, _ = self._get_projection_texture(n)

            self.screen_surface.display(
                tex_id,
                false_colors=self.false_colours,
                uv_encoded_textures=self.uv_encoded_textures
            )

        if view_mode == DisplayMode.Third_person:
            self._render_external_view(point_of_view)

        glBindFramebuffer(GL_FRAMEBUFFER, target_fbo)
        glViewport(0, 0, w, h)

        self._hdr_surface.blit_depth_to(target_fbo)

        self._tonemap_pass()

    def step(self, dt: Optional[float] = None, readback: bool = True) -> Optional['VisualOutput']:
        """
        Advance biological time by one step and render everything.

        batch_size == 1: blocks (via ping-pong PBO) and returns this frame's VisualOutput.
        batch_size > 1: queues the frame on the GPU. Returns None until the batch is full,
            then returns a VisualOutput wrapping the whole batch.

        Args:
            - dt: Optional timestep override (for external control)
            - readback: if False, skip the data download to CPU (time and simulation still advance)
        """

        if self._context is None:
            raise RuntimeError('renderer.step() requires an attached Context.')

        # Sync any CPU-side changes to the eye model
        self.sync_cpu()

        step_dt = dt if dt is not None else self._context.dt

        # Advance dithering for Monte-Carlo noise decorrelation
        if self._time_dithering:
            self._dither_counter += 1

        # Render
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self._eye_uniforms.update(
            dt=step_dt,
            frame_offset=self._frame_index % self._batch_size,
            dither_counter=self._dither_counter,
        )

        self._dispatch()
        self._reduction()
        self._dynamics()

        self._frame_index += 1

        # Grab the output (either single frame or full batch)
        out = None
        if readback:
            if self.runs_interactive or self._batch_size == 1:
                # Depth is paired with the current colour frame using a
                # synchronous read. The old async path returns the previous
                # colour frame and cannot safely be paired with current depth.
                out_array, depth_stats = self._read_outputs_sync()
                raw_depth = self._read_raw_depth_sync() if self._capture_raw_depth else None
                if out_array.size > 0:
                    out = VisualOutput(
                        out_array, self._model,
                        depth=depth_stats['mean'],
                        depth_stats=depth_stats,
                        raw_depth=raw_depth,
                    )
                self._frame_index = 0   # frame consumed, reset counter

            elif self._frame_index >= self._batch_size:
                # Batched path: return full block (only when batch is full)
                out = self.flush()

        self._latest_output = out

        # Collection if history is enabled
        if self._track_history and out is not None:
            self._record_history(out)

        return out

    def set_overlay(self,
            values: Optional[Union[Dict['BaseView', np.ndarray], np.ndarray]] = None,
            range: Optional[Tuple[float, float]] = None,
            colormap: Union[int, 'OverlayColormap'] = OverlayColormap.Thermal,
            compression: float = 1.0,
            autorange_perc: int = 98
        ) -> None:
        """
        Upload data for overlay visualisation.

        Args:
            values: Data to display. Can be:
                - None: Disable overlay and fallback to default (luminance).
                - Array: Global data matching total ommatidia or rhabdomeres.
                - Dict: Mapping of {View: Array}, where Array matches the view's
                        ommatidia or rhabdomere count.
            range: (min, max) bounds. If None, uses automatic scaling.
            colormap: Colormap enum or integer.
            compression: Power exponent for dynamic range compression (gamma).
            autorange_perc: Percentile used for automatic range scaling.
        """

        model_size = self._model.size

        # Ensure overlay SSBO is allocated and sized for current model
        if 'overlay' not in self.eye_buffers or self.eye_buffers['overlay'].count != model_size:
            self.eye_buffers.allocate('overlay', dtype=np.float32, count=model_size, usage=GL_DYNAMIC_DRAW)

        if values is None:
            self._eye_uniforms.update(overlay_fallback=True)
            return

        # Put inputs into a flat per-rhab array
        if isinstance(values, dict):
            flat_data = np.zeros(model_size, dtype=np.float32)
            for view, data in values.items():
                data = np.asanyarray(data)

                # if per-ommatidium, expand to rhabdomeres
                if data.size == view.N:
                    data = np.repeat(data, view.R)

                if data.size != view.size:
                    raise ValueError(f'Overlay mismatch for {view}: got {data.size} elements, '
                                     f'expected {view.N} (ommatidia) or {view.size} (rhabdomeres).')

                flat_data[view.rhab_indices.ravel()] = data.ravel()
            values = flat_data

        else:
            values = np.asanyarray(values, dtype=np.float32).ravel()
            # Handle model-wide expansion if per-ommatidium data is provided
            if values.size == self._model.N:
                values = np.repeat(values, self._model.R)

            if values.size != model_size:
                raise ValueError(f'Global overlay size {values.size} mismatch (expected {model_size}).')

        self.eye_buffers['overlay'].write(values)

        # Range scaling
        self._overlay_autorange_perc = autorange_perc
        if range is not None:
            self._overlay_range = (float(range[0]), float(range[1]))
        else:
            # Auto-range based on percentile to reject outliers
            frame_peak = float(np.percentile(np.abs(values), self._overlay_autorange_perc))

            # Asymmetric EMA: adapt quickly to signal rises, slowly to drops
            if getattr(self, '_overlay_current_peak', 0.0) <= 0.0:
                self._overlay_current_peak = frame_peak

            alpha = 0.5 if frame_peak > self._overlay_current_peak else 0.02
            self._overlay_current_peak = (1.0 - alpha) * self._overlay_current_peak + alpha * frame_peak

            bound = max(self._overlay_current_peak, 1e-6)
            self._overlay_range = (-bound, bound) if colormap == OverlayColormap.Diverging else (0.0, bound)

        self._overlay_colormap = to_enum(colormap, OverlayColormap)
        self._overlay_compression = compression

        self._eye_uniforms.update(
            overlay_fallback=False,
            overlay_data_min=self._overlay_range[0],
            overlay_data_max=self._overlay_range[1],
            overlay_colormap=int(self._overlay_colormap),
            overlay_compression=self._overlay_compression,
        )

    def take_snapshot(self,
            filepath: Union[str, 'Path'],
            view_mode: Union[str, 'DisplayMode'],
            point_of_view: Union['Agent', 'OrbitCamera'],
            width: Optional[int] = None,
            height: Optional[int] = None,
            transparent: bool = True
            ) -> None:
        """
        Renders the current view to an off-screen buffer and saves it as a transparent PNG.
        """
        from PIL import Image

        w = width or self.context.viewport_size[0]
        h = height or self.context.viewport_size[1]

        # Save current state
        prev_fbo = glGetIntegerv(GL_DRAW_FRAMEBUFFER)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)

        bg_alpha = 0.0 if transparent else 1.0
        self._snapshot_target.bind(w, h)
        self.draw(view_mode, point_of_view, target_fbo=self._snapshot_target.fbo_id,
                  override_size=(w, h), bg_alpha=bg_alpha)

        # Read back pixels
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE)

        # Restore state
        self._snapshot_target.unbind()
        glBindFramebuffer(GL_FRAMEBUFFER, prev_fbo)
        glViewport(*prev_viewport)

        # Save
        image = Image.frombytes('RGBA', (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)
        filepath = str(filepath)
        if not filepath.lower().endswith('.png'):
            filepath += '.png'
        image.save(filepath)

        print(f'Snapshot saved to {filepath}')

    @property
    def latest_output(self) -> Optional['VisualOutput']:
        return self._latest_output

    @property
    def history(self) -> Optional['VisualOutput']:
        """
        Returns the full history of rendered frames as a single memmap-backed VisualOutput
        (frames are never held in RAM).
        """
        # Drain GPU pipe of any leftover frames
        remainder = self.flush()
        if remainder is not None:
            self._record_history(remainder)

        if self._history is None or len(self._history) == 0:
            return None

        return VisualOutput(self._history.array, self._model, coords='rhabdomeres', is_time_series=True)

    def clear_history(self) -> None:
        """Discards recorded history and frees the backing file."""
        if self._history is not None:
            self._history.close()
            self._history = None
        self._frame_index = 0

    @property
    def history_path(self) -> Optional[Path]:
        """Backing file for history recording (None = OS temp file, removed on clear_history)."""
        return self._history_path

    @history_path.setter
    def history_path(self, path: Optional[Union[str, Path]]) -> None:
        if self._history is not None:
            raise RuntimeError('History is already being recorded. Call clear_history() before changing history_path.')
        self._history_path = Path(path) if path is not None else None

    # Public properties and methods

    @property
    def model(self) -> 'Model':
        return self._model

    @model.setter
    def model(self, new_model: 'Model') -> None:
        if new_model is self._model:
            return
        same_topology = (new_model.shape == self._model.shape)
        self._model = new_model

        if same_topology:
            # Size unchanged: buffer sizes still valid
            # Re-upload static+dynamic rows and refresh model-derived uniforms
            self.sync_cpu(force_all=True)
            self._update_model_uniforms()
        else:
            # Topology changed: rebuild eye-side GPU resources
            self._free_model_resources()
            self._batch_size, self._samples_per_rhab = self._safe_samples_lim(
                self._batch_size, self._samples_per_rhab, prioritize_batch=True
            )
            self._init_model_resources()  # reallocates SSBOs/PBOs, recompiles, pushes uniforms

        # Selection + overlay buffers are sized by `size`, reset them
        self._selected_omm_indices.fill(-1)
        self._update_selected_ommatidia()

    @property
    def scene(self) -> 'Scene':
        return self._scene

    @scene.setter
    def scene(self, new_scene: 'Scene') -> None:

        if new_scene is self._scene:
            return

        self._scene = new_scene
        self._baker.free()
        self._baker = SceneBaker(new_scene, self._resource_manager)
        self._invalidate_shaders()  # light-count #defines might have changed

        using_sky = new_scene.sky is not None
        self._scene_uniforms.update(
            nb_tlas_nodes=len(self._baker.cpu_tlas_nodes),
            background_color=new_scene.background_color,
            use_sky=using_sky,
            sky_texture=self._baker.scene_textures['sky_texture'].unit if using_sky else 0,
            sh_irradiance_coeffs=(new_scene.sky.sh_coeffs if using_sky
                                  else constant_sh(new_scene.background_color)),
        )
        self._lights_uniforms.update(
            directional_lights_count=self._baker._nb_dir_lights,
            point_lights_count=self._baker._nb_point_lights,
            area_lights_count=self._baker._nb_area_lights,
        )

    @property
    def context(self) -> Optional['Context']:
        """The Context this renderer is attached to."""
        return self._context

    @property
    def screen_surface(self) -> Optional['TextureViewer']:
        if self._screen_surface is None:
            self._screen_surface = TextureViewer()
        return self._screen_surface

    @property
    def exposure(self) -> float:
        return self._exposure

    @exposure.setter
    def exposure(self, value: float) -> None:
        self._exposure = float(value)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:

        new_batch = max(1, int(value))
        if new_batch == self._batch_size:
            return

        # Important: flush any partial batch before resizing buffers
        if self._frame_index > 0:
            leftover = self.flush()
            if self._track_history and leftover:
                self._record_history(leftover)

        # Recalculate safe limits (VRAM check)
        safe_batch, _ = self._safe_samples_lim(new_batch, self._samples_per_rhab)
        self._batch_size = safe_batch

        # Resize colours SSBO
        self.eye_buffers['colors'].resize(self._model.size * self._batch_size)

        # Reallocate PBOs
        self._pbo_index = 0
        for i in range(2):
            self.eye_buffers.allocate(f'pbo_{i}',
                                      dtype=np.uint8,
                                      count=self._model.size * 16,
                                      target=GL_PIXEL_PACK_BUFFER,
                                      usage=GL_STREAM_READ)

        print(f'Renderer batch size updated to: {self._batch_size}')

    @property
    def nb_samples(self) -> int:
        return self._samples_per_rhab

    @nb_samples.setter
    def nb_samples(self, value: int)  -> None:
        _, safe_samples = self._safe_samples_lim(self._batch_size, value, prioritize_batch=False)
        if safe_samples == self._samples_per_rhab:
            return

        self._samples_per_rhab = safe_samples
        rays_elements = self._model.N if self._model.bundle.fused_rhabdoms else self._model.size
        self.eye_buffers['rays_intermediate'].resize(rays_elements * self._samples_per_rhab)

        # Update noise thresholds and uniforms
        mc_noise = 0.65 / np.sqrt(max(1, self._samples_per_rhab))
        self._noise_threshold = max(0.05, mc_noise)
        self._eye_uniforms.update(nb_samples=self._samples_per_rhab, noise_threshold=self._noise_threshold)

        if self._collect_defines() != self._current_defines:
            self._invalidate_shaders()

    @property
    def pixel_samples(self) -> int:
        """Rays per pixel for the third-person / external ray-traced views."""
        return self._samples_per_px

    @pixel_samples.setter
    def pixel_samples(self, value: int) -> None:
        self._samples_per_px = max(1, int(value))
        self._eye_uniforms.update(pixel_samples=self._samples_per_px)

    @property
    def max_bounces(self) -> int:
        """Number of path-tracing bounces. 0 means standard single-bounce raytracing."""
        return self._max_bounces

    @max_bounces.setter
    def max_bounces(self, value: int) -> None:
        val = max(0, int(value))
        if val == self._max_bounces:
            return

        was_pt = self.path_tracing
        new_is_pt = val > 0
        self._max_bounces = val

        self._scene_uniforms.update(max_bounces=self._max_bounces)

        # If toggled between raytracing and pathtracing the defines changed,
        # so shaders must be invalidated and recompiled
        if was_pt != new_is_pt:
            self._current_defines = self._collect_defines()
            self._invalidate_shaders()

    @property
    def ray_tracing(self)-> bool:
        return self._max_bounces == 0

    @property
    def path_tracing(self) -> bool:
        return self._max_bounces > 0

    @property
    def output_mode(self) -> 'EyeOutput':
        return self._output_mode

    @output_mode.setter
    def output_mode(self, value) -> None:
        self._output_mode = value
        self._eye_uniforms.update(output_mode=self._output_mode)
        self._update_selected_ommatidia()

    @property
    def projection_mode(self) -> 'OmmatidiaProjection':
        return self._projection_mode

    @projection_mode.setter
    def projection_mode(self, value) -> None:
        self._projection_mode = value
        self._eye_uniforms.update(projection_mode=self._projection_mode)

    @property
    def tiled_mode(self) -> bool:
        return self._tiled_mode

    @tiled_mode.setter
    def tiled_mode(self, value: bool) -> None:
        self._tiled_mode = bool(value)
        self._eye_uniforms.update(tiled_mode=self._tiled_mode)

    @property
    def hybrid_sampling(self) -> bool:
        """Toggle between Importance Sampling (False) and Hybrid Weighted Sampling (True)."""
        return self._use_hybrid_sampling

    @hybrid_sampling.setter
    def hybrid_sampling(self, value: bool) -> None:
        self._use_hybrid_sampling = bool(value)
        self._eye_uniforms.update(use_hybrid_sampling=self._use_hybrid_sampling)

    def _get_waveguide_lut(self) -> np.ndarray:
        """
        Per-rhabdomere-type angular sensitivity profiles for SamplingMode.Waveguide.

        Falls back to Gaussians if the bundle has no focal length (no F-number, so no mode coupling to compute)
        """
        bundle = self._model.bundle

        if bundle.focal_um is None:
            return waveguide_sensitivity_lut([], [], 1.0)

        apertures = np.asarray(self._model.buffer['aperture_um'], dtype=np.float64)
        f_number = float(bundle.focal_um / max(np.median(apertures), 1e-6))

        return waveguide_sensitivity_lut(
            diameters_um=bundle.diameters_um,
            wavelengths_um=np.asarray(bundle.wavelengths_nm, dtype=np.float64) * 1e-3,
            f_number=f_number,
            n_rhabdomere=bundle.n_rhabdomere,
            n_surround=bundle.n_surround,
            defocus_um=bundle.tip_defocus_um,
            focal_um=bundle.focal_um,
        )

    @property
    def sampling_mode(self) -> 'SamplingMode':
        """The sensitivity profile used for weighting: 'gaussian', 'airy' or 'waveguide'."""
        return self._sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, value: Union[int, str, 'SamplingMode']) -> None:
        self._sampling_mode = to_enum(value, SamplingMode)
        self._eye_uniforms.update(sampling_mode=int(self._sampling_mode))

        if self._sampling_mode is SamplingMode.Waveguide and not self._use_hybrid_sampling:
            self.hybrid_sampling = True
            print('Sampling Mode: Waveguide. Enabling hybrid sampling.')

        print(f"Sampling Mode: {self._sampling_mode.name}")

    @property
    def model_exaggeration(self) -> float:
        return self._eyes_exaggeration

    @model_exaggeration.setter
    def model_exaggeration(self, value: float) -> None:
        self._eyes_exaggeration = float(value)
        self._eye_uniforms.update(**self._update_visualisation_scales())

    @property
    def receptive_field_exaggeration(self) -> float:
        return self._rf_exaggeration

    @receptive_field_exaggeration.setter
    def receptive_field_exaggeration(self, value: float) -> None:
        self._rf_exaggeration = float(value)
        self._eye_uniforms.update(**self._update_visualisation_scales())

    @property
    def saccade_exaggeration(self) -> float:
        return self._saccade_exaggeration

    @saccade_exaggeration.setter
    def saccade_exaggeration(self, value: float) -> None:
        self._saccade_exaggeration = float(value)
        self._eye_uniforms.update(**self._update_visualisation_scales())

    @property
    def reference_luminance(self) -> float:
        return self._lum_ref

    @reference_luminance.setter
    def reference_luminance(self, value) -> None:
        self._lum_ref = float(value)
        self._eye_uniforms.update(lum_ref=self._lum_ref)

    @property
    def noise_threshold(self) -> float:
        return self._noise_threshold

    @noise_threshold.setter
    def noise_threshold(self, value: float) -> None:
        self._noise_threshold = float(value)
        self._eye_uniforms.update(noise_threshold=self._noise_threshold)

    @property
    def clip_ratio(self) -> float:
        """RF clipping at full rhabdomere-aperture offset (1.0 = pure optics)."""
        return float(self._model.bundle.clip_ratio)

    @clip_ratio.setter
    def clip_ratio(self, value: float) -> None:
        self._model.bundle.clip_ratio = float(value)
        self._eye_uniforms.update(clip_ratio=float(self._model.bundle.clip_ratio))

    @property
    def aperture_follow(self) -> float:
        """Swing effect: how much of the lateral move the cone/pigment aperture follows."""
        return float(self._model.bundle.aperture_follow)

    @aperture_follow.setter
    def aperture_follow(self, value: float) -> None:
        self._model.bundle.aperture_follow = float(min(max(0.0, value), 1.0))
        self._eye_uniforms.update(aperture_follow=float(self._model.bundle.aperture_follow))

    @property
    def trigger_delay(self) -> float:
        """Onset latency of the photomechanical arc (s)."""
        return self._trigger_delay

    @trigger_delay.setter
    def trigger_delay(self, value: float) -> None:
        self._trigger_delay = float(value)
        self._eye_uniforms.update(trigger_delay=self._trigger_delay)

    @property
    def time_dithering(self) -> bool:
        return self._time_dithering

    @time_dithering.setter
    def time_dithering(self, value: bool) -> None:
        self._time_dithering = bool(value)
        print(f"Time dithering: {'Enabled' if self._time_dithering else 'Disabled'}.")

    @property
    def randomness_mode(self) -> 'RandomnessMode':
        """The randomness mode used for sampling: 'pseudorandom', 'halton' or 'stratified'."""
        return self._randomness_mode

    @randomness_mode.setter
    def randomness_mode(self, value: Union[int, str, RandomnessMode]) -> None:
        self._randomness_mode = to_enum(value, RandomnessMode)
        self._eye_uniforms.update(randomness_mode=int(self._randomness_mode))
        print(f"Randomness Mode: {self._randomness_mode.name}")

    @property
    def photon_concentration(self) -> float:
        return float(self._eye_uniforms['photon_concentration_factor'])

    @photon_concentration.setter
    def photon_concentration(self, value: float) -> None:
        self._eye_uniforms.update(photon_concentration_factor=float(value))

    @property
    def overlay_enabled(self) -> bool:
        return self._overlay_enabled and 'overlay' in self.eye_buffers and self.eye_buffers['overlay'].count > 0

    @overlay_enabled.setter
    def overlay_enabled(self, value: bool) -> None:
        enable = bool(value)
        self._overlay_enabled = enable
        if enable:
            self._eye_uniforms.update(
                overlay_fallback=True,
                overlay_data_min=0.0,
                overlay_data_max=1.0,
                overlay_colormap=int(OverlayColormap.Thermal),
                overlay_compression=1.0
            )
            # dummy call to initialise the buffer
            self.set_overlay()

    @property
    def false_colours(self) -> bool:
        return self._false_colours

    @false_colours.setter
    def false_colours(self, v: bool) -> None:
        self._false_colours = bool(v)
        self._eye_uniforms.update(false_colors=self._false_colours and not self.uv_encoded_textures)

    @property
    def selected_ommatidia(self) -> Optional[list]:
        active = [int(x) for x in self._selected_omm_indices if x != -1]
        return active if active else None

    @selected_ommatidia.setter
    def selected_ommatidia(self, values: Optional[Union[int, Sequence[int], np.ndarray]]) -> None:
        self._selected_omm_indices.fill(-1)

        if values is not None:
            if isinstance(values, np.ndarray):
                vals = values.ravel()
                count = min(len(vals), 10)
                self._selected_omm_indices[:count] = vals[:count].astype(np.int32)
            elif isinstance(values, int):
                self._selected_omm_indices[0] = values
            else:
                for i, val in enumerate(list(values)[:10]):
                    self._selected_omm_indices[i] = int(val)

        self._update_selected_ommatidia()

    @property
    def microsaccades_enabled(self) -> bool:
        return self._microsaccades_enabled

    @microsaccades_enabled.setter
    def microsaccades_enabled(self, value: bool) -> None:
        value = bool(value)
        if value and not self._model.has_microsaccades:
            print("Can't enable microsaccades. This eye model has 0.0 microsaccade amplitude.")
            value = False
        self._microsaccades_enabled = value
        self._eye_uniforms.update(enable_actuation=self._microsaccades_enabled)

    @property
    def pupil_drive(self) -> Optional[float]:
        """
        Overrides the steady-state drive with a fixed value in [0, 1].
        None disables the override.
        """
        return self._override_pupil_drive

    @pupil_drive.setter
    def pupil_drive(self, value: Optional[float]) -> None:
        self._override_pupil_drive = None if value is None else float(value)

        is_overriden = self._override_pupil_drive is not None

        self._eye_uniforms.update(
            force_pupil_drive=is_overriden,
            pupil_drive=self._override_pupil_drive if is_overriden else 0.0,
        )

    @property
    def saccade_drive(self) -> Optional[float]:
        """
        Overrides the photomechanical reflex arc with a fixed excursion in [0, 1].
        None disables the override.
        """
        return self._override_saccade_drive

    @saccade_drive.setter
    def saccade_drive(self, value: Optional[float]) -> None:
        self._override_saccade_drive = None if value is None else float(value)

        is_overriden = self._override_saccade_drive is not None

        self._eye_uniforms.update(
            force_saccade_drive=is_overriden,
            saccade_drive=self._override_saccade_drive if is_overriden else 0.0,
        )

    def dither(self) -> None:
        """Dither once (reshuffle the dither counter)"""
        self._dither_counter = random.randint(0, 1024)

    dither_once = dither

    @property
    def tlas(self) -> Optional['BVH']:
        return self._baker.tlas

    @property
    def blases(self) -> List['BVH']:
        return self._baker.blases

    @property
    def sky_intensity(self) -> float:
        return self._sky_intensity

    @sky_intensity.setter
    def sky_intensity(self, value) -> None:
        self._sky_intensity = float(value)
        self._lights_uniforms.update(sky_intensity=self._sky_intensity)

    @property
    def ambient_intensity(self) -> float:
        return self._ambient_intensity

    @ambient_intensity.setter
    def ambient_intensity(self, value) -> None:
        self._ambient_intensity = float(value)
        self._lights_uniforms.update(ambient_intensity=self._ambient_intensity)

    @property
    def enable_ambient(self) -> bool:
        return self._enable_ambient

    @enable_ambient.setter
    def enable_ambient(self, value) -> None:
        self._enable_ambient = bool(value)
        self._lights_uniforms.update(enable_ambient=self._enable_ambient)

    @property
    def enable_direct(self):
        return self._enable_direct

    @enable_direct.setter
    def enable_direct(self, value) -> None:
        self._enable_direct = bool(value)
        self._lights_uniforms.update(enable_direct=self._enable_direct)

    @property
    def enable_shadows(self) -> bool:
        return self._enable_shadows

    @enable_shadows.setter
    def enable_shadows(self, value) -> None:
        self._enable_shadows = bool(value)
        self._lights_uniforms.update(enable_shadows=self._enable_shadows)

    # TODO: Reorder the properties

    # Cleanup

    def free(self) -> None:
        """
        Free all GPU resources owned by the renderer.
        """

        # Recorded history (releases the memmap and deletes the backing file)
        self.clear_history()

        # Eyes stuff: shaders, SSBOs/PBOs, fences, eye-mesh VAOs/shaders
        self._free_model_resources()

        # Scene baking (BVH, lights, scene textures)
        self._baker.free()

        # View-side render targets and the fullscreen-blit helper
        self._projection_textures.free()
        if self._screen_surface:
            self._screen_surface.free()
            self._screen_surface = None

        # Presentation: HDR target + tonemap pass (moved in from Context)
        self._hdr_surface.free()
        if self._tonemap_shader:
            self._tonemap_shader.free()
            self._tonemap_shader = None

        if self._tonemap_vao:
            glDeleteVertexArrays(1, [self._tonemap_vao])
            self._tonemap_vao = None

        self._snapshot_target.free()