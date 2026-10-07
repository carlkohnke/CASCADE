# Custom geometry format

CASCADE accepts explicit vascular segment networks in CSV or NPZ format through `network.mode="simple"` and `network.simple.mode="custom"`.

## CSV

Columns, in centimetres:

```text
start_x,start_y,start_z,end_x,end_y,end_z,radius_cm,prox_id,dist_id
```

Make sure the columns have headers. Coordinates and `radius_cm` are required; only `prox_id` and `dist_id` are optional. When node IDs are absent, coincident endpoints are matched after rounding coordinates to 12 decimal places.

## NPZ

Arrays:

- `starts`: shape `(n_segments, 3)`
- `ends`: shape `(n_segments, 3)`
- `radii`: shape `(n_segments,)`
- `prox_ids`: shape `(n_segments,)`
- `dist_ids`: shape `(n_segments,)`

Coordinates and radii are in centimetres. Node IDs are non-negative integers. Every segment must have positive length and radius.

Source-only graph nodes are inferred as inlets and sink-only nodes as outlets. Override them with `network.simple.inlet_nodes` and `network.simple.outlet_nodes` when graph degree is not sufficient, such as a recirculating network.
