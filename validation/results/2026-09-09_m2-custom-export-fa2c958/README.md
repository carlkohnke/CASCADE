# Custom geometry, domain, and normal-export certification

Tracker IDs: VAL-06 (normal-export half) and VAL-09. Candidate: exact installed wheel from commit `fa2c958`, SHA-256 `de21614b728dd3963db83176e3c7c99ed45b8f2342ea83a6c59985f24565f8cf`.

All commands ran from `/tmp`, outside the source checkout, while settings and input paths resolved relative to their settings files or through the installed resource resolver.

| Case | Network input | Domain input | Result | Export checks |
|:---|:---|:---|:---:|---:|
| CUSTOM-Y | CSV, SHA `b36f0812...` | built-in box | pass | 23/23 |
| CUSTOM-NPZ | NPZ, SHA `c1f6a9c6...` | built-in box | pass | 23/23 |
| CUSTOM-DOMAIN-STL | CSV, SHA `db2e0d9b...` | installed `bivent3.stl`, SHA `10fd4976...` | pass | 23/23 |
| CUSTOM-DOMAIN-VTP | NPZ, SHA `c1f6a9c6...` | VTP, SHA `c5f2f145...` | pass | 23/23 |

Each inspection reopened `vessels.vtp`, `oxygen_points.vtp`, `domain_boundary.vtp`, and `domain_mesh.vtu`; required nonempty datasets; inventoried point/cell fields and dtypes; required one vessel cell per segment and complete global IDs; and compared flow, pressure, radius, length, point IDs, and local tissue concentration against CSV output.

CSV and NPZ representations of the same Y network produced byte-identical `segments.csv` and `points.csv`, and all 14 available scientific summary fields compared exactly. The first CUSTOM-Y inspection is retained as a failed checker attempt because it expected a nonexistent `tissue_oxygen` field; `local_concentration` is the actual CASCADE contract. `custom-y-export-inspection-v2.json` is the corrected passing evidence.

Heart-specific VTK/Cext export remains under VAL-06/VAL-07 and is not certified by this bounded normal-export campaign.
