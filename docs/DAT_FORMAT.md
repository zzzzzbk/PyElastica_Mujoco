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
    }
  ],
  "fields": { ... optional packed arrays ... },
  "metadata": { ... }
}
```

Each packed array is a dictionary containing `dtype`, `shape`, and raw `data`
bytes. This avoids pickling NumPy implementation classes and makes files
portable across different NumPy builds.

All rods share the same strictly increasing frame times but may have different
element counts. `radius` is the static element-radius profile used for
visualization. `fields` can hold application-specific arrays such as muscle
activation without coupling the renderer to them.

Readers retain compatibility with unversioned callback DAT files containing
raw/list-valued histories under `systems`. Because DAT is a pickle container,
only load files from trusted sources.
