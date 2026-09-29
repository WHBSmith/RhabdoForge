import json
import argparse
from pathlib import Path
import numpy as np
import polars as pl

from rhabdoforge.engine import get_context, Agent, Scene, Asset
from rhabdoforge.compound_eyes import Model
from rhabdoforge.compound_eyes.rhabdomeres import drosophila_bundle
from rhabdoforge.compound_eyes.helpers.waveguide import WaveguideAcceptance
from rhabdoforge.renderers import Renderer
from rhabdoforge.types import RandomnessMode, SamplingMode


# Paths are anchored to this repository rather than the process working
# directory, so the recorder also works when launched from another project.
RHABDOFORGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ENV_DIR = RHABDOFORGE_ROOT / 'assets' / 'environments'
PROJECT_OUTPUT_DIR = RHABDOFORGE_ROOT / 'data' / 'generated'
MODEL_ASSET = RHABDOFORGE_ROOT / 'assets' / 'drosophila_scaffold.npz'
SKY_ASSET = RHABDOFORGE_ROOT / 'assets' / 'textures' / 'kloppenheim_05_4k.exr'

# Config
# See record_dataset.md

ENVIRONMENT = 'seville'        # 'seville' or 'canberra'
ENABLE_DYNAMICS = False        # microsaccades + pupil adaptation (pupil forced halfway when on)
SAMPLES_PER_RHABDOMERE = 64
OUTPUT_DIR = PROJECT_OUTPUT_DIR
RECORD_KEY = 'f9'

PUPIL_DRIVE = 0.5
SCRIPTED_STEPS = 1000
SCRIPTED_SPEED = 0.5
SCRIPTED_RADIUS = 3.0

ENV_ASSETS = {
    'seville': PROJECT_ENV_DIR / 'seville_filtered.ply',
    'canberra': PROJECT_ENV_DIR / 'canberra_filtered.ply',
}

# -----------------------------------------------


def save_take(rows: list, layout: dict, meta: dict) -> None:

    if not rows:
        print('[record] nothing to save (empty take)')
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame(rows).with_columns(
        pl.col('visual_output').cast(pl.List(pl.Float32)),
        pl.col('depth').cast(pl.List(pl.Float32)),
    )
    df.write_parquet(OUTPUT_DIR / 'data.parquet')

    np.savez(OUTPUT_DIR / 'layout.npz', **layout)

    with open(OUTPUT_DIR / 'metadata.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'[record] saved {len(rows)} frames -> {OUTPUT_DIR}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--mode', choices=('interactive', 'scripted'), default='interactive',
        help='Record manually or run a reproducible scripted trajectory.',
    )
    parser.add_argument(
        '--path', choices=('straight', 'circle'), default='straight',
        help='Scripted path type.',
    )
    parser.add_argument('--steps', type=int, default=SCRIPTED_STEPS)
    parser.add_argument('--speed', type=float, default=SCRIPTED_SPEED,
                        help='Straight-line speed in metres per second.')
    parser.add_argument('--radius', type=float, default=SCRIPTED_RADIUS,
                        help='Radius of the circular path in metres.')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        '--environment-file', type=Path, default=None,
        help='Optional explicit PLY path; overrides the selected environment asset.',
    )
    return parser.parse_args()


def advance_scripted_path(agent: Agent, path: str, step: int, dt: float,
                          speed: float, radius: float, start_position) -> None:
    """Advance one deterministic scripted trajectory step."""
    if path == 'straight':
        agent.translate(agent.forward * speed * dt)
        return

    # Circle in the X/Z plane, looking toward the centre. This uses the same
    # world coordinates as the Seville/Canberra assets.
    centre = np.asarray(start_position, dtype=np.float32)
    theta = step * speed * dt / max(radius, 1e-6)
    position = centre + np.array([radius * np.sin(theta), 0.0,
                                  radius * np.cos(theta)], dtype=np.float32)
    agent.position = position
    agent.lookat(centre)


if __name__ == '__main__':

    args = parse_args()
    if args.steps < 1:
        raise ValueError('--steps must be positive')
    OUTPUT_DIR = args.output_dir.expanduser().resolve()
    environment_path = (
        args.environment_file.expanduser().resolve()
        if args.environment_file is not None
        else ENV_ASSETS[ENVIRONMENT]
    )

    if not MODEL_ASSET.is_file():
        raise FileNotFoundError(f'Model asset not found: {MODEL_ASSET}')
    if not SKY_ASSET.is_file():
        raise FileNotFoundError(f'Sky asset not found: {SKY_ASSET}')
    if not environment_path.is_file():
        raise FileNotFoundError(f'Environment asset not found: {environment_path}')

    context = get_context()

    scene = Scene(background_color=(0.15, 0.15, 0.3))
    scene.add_sky(str(SKY_ASSET))

    env_asset = Asset.from_file(name=ENVIRONMENT, file_path=str(environment_path))
    scene.add_instance(env_asset)

    model = Model.from_file(
        str(MODEL_ASSET),
        bundle=drosophila_bundle(),
        acceptance=WaveguideAcceptance(),
        neural_superposition=True,
    )
    model.scale(1e-6)

    agent = Agent(position=(0.0, 0.0, 4.0))

    renderer = Renderer(
        model=model, scene=scene, agent=agent,
        nb_samples=SAMPLES_PER_RHABDOMERE,
        time_dithering=True,
        randomness_mode=RandomnessMode.Halton,
        sampling_mode=SamplingMode.Waveguide,
        enable_microsaccades=ENABLE_DYNAMICS,
        enable_direct=True, enable_shadows=True, enable_ambient=True,
    )
    renderer.hybrid_sampling = True
    renderer.pupil_drive = PUPIL_DRIVE if ENABLE_DYNAMICS else None

    # Cartridge (post neural-superposition) layout: saved with every take so a
    # frame's flat visual_output array can be placed back in visual space without needing rhabdoforge
    cart = model.ommatidia.cartridges
    N, R = model.shape
    layout = {
        'N': N,
        'R': R,
        'azimuth': np.asarray(cart.rhabdomere_azimuth, dtype=np.float32),      # (N, R) rad
        'elevation': np.asarray(cart.rhabdomere_elevation, dtype=np.float32),  # (N, R) rad
    }

    meta = {
        'environment': ENVIRONMENT,
        'dynamics_enabled': ENABLE_DYNAMICS,
        'pupil_drive': renderer.pupil_drive,
        'nb_samples_per_rhabdomere': SAMPLES_PER_RHABDOMERE,
        'sampling_mode': 'Waveguide',
        'model_scale': 1e-6,
        'visual_output_channels': ['R', 'G', 'B', 'gain'],
        'visual_output_shape': [N, R, 4],
        'depth_shape': [N, R],
        'depth_units': 'metres',
        'depth_definition': 'mean valid primary-ray hit distance',
        'depth_miss_value': 'NaN',
    }

    recording = args.mode == 'scripted'
    rows = []
    start_position = tuple(float(v) for v in agent.position)

    def toggle_recording():
        global recording
        recording = not recording
        if recording:
            rows.clear()
            print('[record] started')
        else:
            print(f'[record] stopped ({len(rows)} frames)')
            save_take(rows, layout, meta)

    context.bind_key(RECORD_KEY, toggle_recording, description='Toggle recording')

    print(f"Environment: {ENVIRONMENT} | dynamics: {'on' if ENABLE_DYNAMICS else 'off'}")
    if args.mode == 'interactive':
        print(f'Fly around with WASD + mouse. Press {RECORD_KEY.upper()} to start/stop recording.')
    else:
        print(f'Scripted path: {args.path} | steps: {args.steps}')

    if args.mode == 'scripted':
        for step, dt in enumerate(context.run_headless(steps=args.steps)):
            advance_scripted_path(agent, args.path, step, dt, args.speed,
                                  args.radius, start_position)
            output = renderer.step(dt)
            if output is not None:
                pos = agent.position
                gaze = agent.forward
                rows.append({
                    'frame': context.frame_count,
                    't': context.total_time,
                    'pos_x': float(pos.x), 'pos_y': float(pos.y), 'pos_z': float(pos.z),
                    'yaw': float(agent.yaw), 'pitch': float(agent.pitch), 'roll': float(agent.roll),
                    'gaze_x': float(gaze.x), 'gaze_y': float(gaze.y), 'gaze_z': float(gaze.z),
                    'visual_output': output.per_cartridge.data.reshape(-1).tolist(),
                    'depth': np.asarray(output.depth, dtype=np.float32).reshape(-1).tolist(),
                })
        save_take(rows, layout, meta)
        context.free()
        raise SystemExit

    while context.run_interactive(use_dashboard=True):

        context.input()

        output = renderer.step()

        if recording and output is not None:
            pos = agent.position
            gaze = agent.forward
            rows.append({
                'frame': context.frame_count,
                't': context.total_time,
                'pos_x': float(pos.x), 'pos_y': float(pos.y), 'pos_z': float(pos.z),
                'yaw': float(agent.yaw), 'pitch': float(agent.pitch), 'roll': float(agent.roll),
                'gaze_x': float(gaze.x), 'gaze_y': float(gaze.y), 'gaze_z': float(gaze.z),
                'visual_output': output.per_cartridge.data.reshape(-1).tolist(),
                'depth': np.asarray(output.depth, dtype=np.float32).reshape(-1).tolist(),
            })

        context.display()

    if recording:
        print(f'[record] window closed mid-take, saving ({len(rows)} frames)')
        save_take(rows, layout, meta)

    context.free()
