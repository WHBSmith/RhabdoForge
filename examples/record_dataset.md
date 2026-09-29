## Record datasets

This uses RhabdoForge's interactive window

Controls are the engine's default:

- WASD to move
- mouse to look
- Space/Ctrl for up/down
- Shift+A/D to strafe
- Q/E to roll

Press **F9**, move wherever you want, press **F9** again to stop

The take is written to disk immediately. You can record as many takes as you want in one session, each one
gets its own timestamped set of files.

## Config

- `ENVIRONMENT`: `'seville'` or `'canberra'`
- `ENABLE_DYNAMICS`: Turns photoreceptor dynamics (microsaccades + pupil adaptation) on or off **[*]**
- `SAMPLES_PER_RHABDOMERE`: How many rays per rhabdomeres
- `RECORD_KEY`: the toggle key, if you want to rebind it
- `PUPIL_DRIVE`: Fixed pupil state (light adaptation). 0.0 for dark adapted, 1.0 for light adapted **[*]**

The recorder also supports reproducible scripted paths. Asset and output paths
are anchored to the project, so it can be launched from any working directory:

```bash
DISPLAY=:0 mamba run -n microDepth python \
	/path/to/micro_depth/external/RhabdoForge/examples/record_dataset.py \
	--mode scripted --path straight --steps 100 --speed 0.5
```

Available scripted paths are `straight` and `circle`. `--steps` controls the
number of frames and `--speed` is metres per second. The default output is
`micro_depth/data/generated`; override it with `--output-dir`. The selected
environment can be overridden with `--environment-file`. The default
`interactive` mode remains unchanged.

> **[*]** When dynamics are enabled, the pupil adaptation state should be fixed (because the dynamics of it are borked currently). Pick whether you want your fly to have better acuity, lower sensitivity (light adapted) or vice versa (dark adapted).

## Output format

Each run generates three files in the selected output directory:

```text
data.parquet
layout.npz
metadata.json
```

### `<take>.parquet`

One row per rendered frame, columns:

| column                    | meaning                                                               |
|---------------------------|-----------------------------------------------------------------------|
| `frame`                   | frame index (from session start, not from take start)                 |
| `t`                       | simulated time (seconds)                                              |
| `pos_x/y/z`               | agent world position (metres)                                         |
| `yaw/pitch/roll`          | agent orientation (degrees)                                           |
| `gaze_x/y/z`              | agent forward direction, unit vector (world space)                    |
| `visual_output`           | flat list of floats, the full per-cartridge eye output for that frame |

`visual_output` reshapes to `(N, R, 4)` where `N` is the number of cartridges, `R` is the number
of rhabdomeres per cartridge (7 for the Drosophila model: R1-R6 peripheral + R7/R8 central), and
the last axis is `(R, G, B, gain)`.

`gain` is the photoreceptor's adaptation state.

### `<take>_layout.npz`

Spatial layout for the cartridges in `visual_output`, so you can place each of the `N * R`
entries in visual space without loading RhabdoForge:

- `N`, `R`: same as above
- `azimuth`, `elevation`: `(N, R)` float32 arrays, radians, world-space viewing direction of  each rhabdomere

### `<take>_meta.json`

Run-level settings that were constant for the take: environment name, whether dynamics
were enabled, forced pupil drive value, sample count, sampling mode, and the model's world
scale