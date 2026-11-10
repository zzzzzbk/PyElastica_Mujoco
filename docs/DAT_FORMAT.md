# PyElastica rod trajectory DAT format

Version 1 stores a trusted-local Python pickle with this structure:

```text
{
  "format": "pyelastica-rod-trajectory",
  "format_version": 1,
  "recording_fps": 30.0,
  "systems": [
    {
      "name": "rod_0",
      "type": "cosserat_rod",
      "time":       packed [frames],
      "position":   packed [frames, 3, nodes],
      "radius":     packed [elements],
      "directors":  packed [frames, 3, 3, elements]  (optional)
    },
    {
      "name": "rigid",
      "type": "rigid_cylinder",
      "time":      packed [frames],
      "position":  packed [frames, 3, 2],  # cylinder endpoints
      "radius":    packed [1]
    }
  ],
  "fields": { ... optional packed arrays ... },
  "metadata": { ... }
}
```

Each packed array is a dictionary containing `dtype`, `shape`, and raw `data`
bytes. This avoids pickling NumPy implementation classes and makes files
portable across different NumPy builds.

All systems share the same strictly increasing frame times but may have different
element counts. `radius` is the static element-radius profile used for
visualization. `fields` can hold application-specific arrays such as muscle
activation without coupling the renderer to them.

Readers retain compatibility with unversioned callback DAT files containing
raw/list-valued histories under `systems`. Older multi-rod callbacks may use a
dictionary keyed by `rod0`, `rod1`, etc. The reader loads those rod entries and
the `rigid` cylinder, and infers playback FPS from frame times
when the file has no `recording_fps`. The viewer uses each rod's first recorded
radius profile because this format represents radius as a static profile. The
rigid cylinder uses its first and last recorded position nodes as endpoints.
Saving one of these loaded files writes the versioned format; legacy-only fields such
as `LK`, `TW`, and per-frame radius changes are not retained. Keep the original
file if those fields matter. Because DAT is a pickle container, only load files
from trusted sources.
