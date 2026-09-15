"""Provide forest persistence, simulation-cache, and multi-tree behavior.

The mixin streams large tree payloads, restores domain/tree relationships, and
builds compact analysis caches without retaining duplicate preallocation arrays.
"""

from __future__ import annotations

import gc
import io
import os
import pickle
import struct
import zipfile

import numpy

from cascade.vessels.metadata import read_npy_header_stream

class _ProgressReader:
    def __init__(self, raw, total, desc):
        self._raw = raw
        self._progress = None
        try:
            from tqdm import tqdm
            self._progress = tqdm(total=int(total), desc=desc, unit="B", unit_scale=True, unit_divisor=1024)
        except Exception:
            self._progress = None

    def _update(self, n_bytes):
        if self._progress is not None and n_bytes:
            self._progress.update(int(n_bytes))

    def read(self, size=-1):
        data = self._raw.read(size)
        self._update(len(data))
        return data

    def readline(self, size=-1):
        data = self._raw.readline(size)
        self._update(len(data))
        return data

    def readinto(self, b):
        n_bytes = self._raw.readinto(b)
        self._update(n_bytes or 0)
        return n_bytes

    def close(self):
        try:
            self._raw.close()
        finally:
            if self._progress is not None:
                self._progress.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getattr__(self, name):
        return getattr(self._raw, name)


class ForestCompatibilityMixin:
    @classmethod
    @staticmethod
    def _resolve_load_mode(mode):
        normalized = str(mode or "growth").strip().lower()
        if normalized not in {"growth", "simulation"}:
            raise ValueError("mode must be 'growth' or 'simulation'.")
        return normalized

    @staticmethod
    def _simulation_preallocation_step(segment_count, fallback_rows):
        loaded_rows = max(int(segment_count), int(fallback_rows), 0)
        return max(loaded_rows + 8, 1)

    @staticmethod
    def _is_simulation_cache(path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with zipfile.ZipFile(path, 'r') as archive:
                return "simulation_meta.pkl" in archive.namelist()
        except zipfile.BadZipFile:
            return False

    @staticmethod
    def _serialize_tree_parameters(tree):
        return {
            'kinematic_viscosity': float(tree.parameters.kinematic_viscosity),
            'fluid_density': float(tree.parameters.fluid_density),
            'terminal_flow': float(tree.parameters.terminal_flow) if tree.parameters.terminal_flow is not None else None,
            'root_flow': float(tree.parameters.root_flow) if tree.parameters.root_flow is not None else None,
            'terminal_pressure': float(tree.parameters.terminal_pressure),
            'root_pressure': float(tree.parameters.root_pressure),
            'murray_exponent': float(tree.parameters.murray_exponent),
            'radius_exponent': float(tree.parameters.radius_exponent),
            'length_exponent': float(tree.parameters.length_exponent),
            'max_nonconvex_count': int(tree.parameters.max_nonconvex_count),
            'unit_system': {
                'length': tree.parameters.unit_system.base.length.symbol,
                'time': tree.parameters.unit_system.base.time.symbol,
                'mass': tree.parameters.unit_system.base.mass.symbol,
                'pressure': tree.parameters.unit_system.pressure.symbol,
            },
        }

    @classmethod
    def _load_simulation_cache(cls, path: str, *, data_dtype=None, index_dtype=None):
        import numpy as np
        from svv.tree.data.data import TreeData, TreeMap
        from svv.tree.data.units import UnitSystem

        resolved_data_dtype = None if data_dtype is None else np.dtype(data_dtype)
        if resolved_data_dtype is not None and resolved_data_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("data_dtype must be float32 or float64.")
        resolved_index_dtype = np.dtype(np.int64 if index_dtype is None else index_dtype)
        if resolved_index_dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
            raise ValueError("index_dtype must be int32 or int64.")

        with zipfile.ZipFile(path, 'r') as archive:
            meta = pickle.loads(archive.read("simulation_meta.pkl"))
            data_members = [
                tree_meta["data_member"]
                for network_meta in meta.get("trees", [])
                for tree_meta in network_meta
                if "data_member" in tree_meta
            ]
            load_progress = None
            restore_progress = None
            try:
                from tqdm import tqdm
                total_bytes = sum(int(archive.getinfo(member).file_size) for member in data_members)
                load_progress = tqdm(
                    total=total_bytes,
                    desc="Loading simulation cache arrays",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                )
                restore_progress = tqdm(
                    total=len(data_members),
                    desc="Restoring simulation trees",
                    unit="tree",
                )
            except Exception:
                load_progress = None
                restore_progress = None

            class _SharedProgressReader:
                def __init__(self, raw, progress):
                    self._raw = raw
                    self._progress = progress

                def read(self, size=-1):
                    data = self._raw.read(size)
                    if self._progress is not None and data:
                        self._progress.update(len(data))
                    return data

                def readline(self, size=-1):
                    data = self._raw.readline(size)
                    if self._progress is not None and data:
                        self._progress.update(len(data))
                    return data

                def readinto(self, b):
                    n_bytes = self._raw.readinto(b)
                    if self._progress is not None and n_bytes:
                        self._progress.update(int(n_bytes))
                    return n_bytes

                def __getattr__(self, name):
                    return getattr(self._raw, name)

            def _read_npy_member(member):
                """Stream physical fields and retain exact integer identifiers.

                SVV stores child, parent, and node identifiers in columns 15:20
                of its floating vessel table. Float32 cannot represent every
                integer above 2**24, so capture those columns from the source
                dtype before downcasting the physical table.
                """
                with archive.open(member, "r") as raw_handle:
                    handle = _SharedProgressReader(raw_handle, load_progress)
                    shape, fortran_order, source_dtype = read_npy_header_stream(
                        handle
                    )
                    source_dtype = np.dtype(source_dtype)
                    if source_dtype.hasobject:
                        raise ValueError(f"Simulation cache member {member!r} contains object data.")
                    target_dtype = resolved_data_dtype or source_dtype
                    if len(shape) != 2 or int(shape[1]) < 20 or fortran_order:
                        raise ValueError(
                            f"Simulation cache member {member!r} must be a C-order vessel table with at least 20 columns."
                        )
                    n_rows, n_cols = (int(shape[0]), int(shape[1]))
                    output = np.empty(shape, dtype=target_dtype, order="C")
                    exact_ids = np.empty((n_rows, 5), dtype=resolved_index_dtype)
                    rows_per_chunk = max(
                        1,
                        (16 * 1024 * 1024) // max(source_dtype.itemsize * n_cols, 1),
                    )
                    for row_offset in range(0, n_rows, rows_per_chunk):
                        row_count = min(rows_per_chunk, n_rows - row_offset)
                        count = row_count * n_cols
                        byte_count = count * source_dtype.itemsize
                        blocks = []
                        remaining = byte_count
                        while remaining:
                            block = handle.read(remaining)
                            if not block:
                                raise ValueError(f"Simulation cache member {member!r} ended before its NPY payload.")
                            blocks.append(block)
                            remaining -= len(block)
                        source = np.frombuffer(
                            b"".join(blocks), dtype=source_dtype, count=count
                        ).reshape((row_count, n_cols))
                        output[row_offset:row_offset + row_count, :] = source
                        raw_ids = source[:, 15:20]
                        finite_ids = np.isfinite(raw_ids)
                        if np.any(finite_ids):
                            finite_values = raw_ids[finite_ids]
                            if np.any(finite_values != np.rint(finite_values)):
                                raise ValueError(
                                    f"Simulation cache member {member!r} contains non-integral topology/node identifiers."
                                )
                            info = np.iinfo(resolved_index_dtype)
                            if float(np.min(finite_values)) < info.min or float(np.max(finite_values)) > info.max:
                                raise OverflowError(
                                    f"Simulation cache member {member!r} identifiers do not fit {resolved_index_dtype}."
                                )
                        exact_chunk = np.full(raw_ids.shape, -1, dtype=resolved_index_dtype)
                        exact_chunk[finite_ids] = raw_ids[finite_ids].astype(
                            resolved_index_dtype, copy=False
                        )
                        exact_ids[row_offset:row_offset + row_count, :] = exact_chunk
                    return output, exact_ids[:, 0:3], exact_ids[:, 3:5]

            forest = cls(
                n_networks=meta.get("n_networks", 1),
                n_trees_per_network=meta.get("n_trees_per_network", [2]),
                start_points=meta.get("start_points"),
                directions=meta.get("directions"),
                physical_clearance=meta.get("physical_clearance", 0.0),
                compete=meta.get("compete", False),
                preallocation_step=1,
            )
            forest.convex = meta.get("convex", None)
            forest.domain = None
            forest.geodesic = None
            forest.connections = None

            for i in range(forest.n_networks):
                for j in range(forest.n_trees_per_network[i]):
                    tree_meta = meta["trees"][i][j]
                    tree = forest.networks[i][j]

                    us_dict = tree_meta["parameters"].get("unit_system", {})
                    unit_system = UnitSystem(
                        length=us_dict.get("length", "cm"),
                        time=us_dict.get("time", "s"),
                        mass=us_dict.get("mass", "g"),
                    )
                    tree.parameters.set_unit_system(unit_system)
                    params = tree_meta["parameters"]
                    tree.parameters.kinematic_viscosity = params["kinematic_viscosity"]
                    tree.parameters.fluid_density = params["fluid_density"]
                    tree.parameters.terminal_flow = params["terminal_flow"]
                    tree.parameters.root_flow = params["root_flow"]
                    tree.parameters.terminal_pressure = params["terminal_pressure"]
                    tree.parameters.root_pressure = params["root_pressure"]
                    tree.parameters.murray_exponent = params["murray_exponent"]
                    tree.parameters.radius_exponent = params["radius_exponent"]
                    tree.parameters.length_exponent = params["length_exponent"]
                    tree.parameters.max_nonconvex_count = params["max_nonconvex_count"]

                    data_array, exact_connectivity, exact_node_ids = _read_npy_member(
                        tree_meta["data_member"]
                    )
                    if "connectivity_member" in tree_meta:
                        with archive.open(tree_meta["connectivity_member"], "r") as handle:
                            exact_connectivity = np.asarray(
                                np.load(handle, allow_pickle=False), dtype=resolved_index_dtype
                            )
                    if "node_ids_member" in tree_meta:
                        with archive.open(tree_meta["node_ids_member"], "r") as handle:
                            exact_node_ids = np.asarray(
                                np.load(handle, allow_pickle=False), dtype=resolved_index_dtype
                            )
                    if data_array.ndim != 2:
                        data_array = np.atleast_2d(data_array)
                    n_rows = int(tree_meta.get("segment_count", data_array.shape[0]))
                    if n_rows <= 0:
                        n_rows = int(data_array.shape[0])
                    if n_rows < data_array.shape[0]:
                        data_array = data_array[:n_rows, :]
                    if exact_connectivity.ndim != 2 or exact_connectivity.shape[0] < n_rows or exact_connectivity.shape[1] != 3:
                        raise ValueError(
                            f"Simulation cache tree {i}/{j} has invalid connectivity shape {exact_connectivity.shape}."
                        )
                    if exact_node_ids.ndim != 2 or exact_node_ids.shape[0] < n_rows or exact_node_ids.shape[1] != 2:
                        raise ValueError(
                            f"Simulation cache tree {i}/{j} has invalid node-id shape {exact_node_ids.shape}."
                        )

                    tree_data = TreeData.from_array(np.asarray(data_array))
                    tree.data = tree_data
                    tree.data_dtype = np.dtype(data_array.dtype)
                    tree.index_dtype = resolved_index_dtype
                    tree.segment_count = n_rows
                    tree.preallocation_step = n_rows
                    tree.preallocate = tree.data
                    tree.preallocate_midpoints = np.empty((0, 3), dtype=tree.data_dtype)
                    tree.midpoints = tree.preallocate_midpoints
                    tree.connectivity = exact_connectivity[:n_rows]
                    tree._cascade_node_ids = exact_node_ids[:n_rows]
                    tree.vessel_map = TreeMap()
                    tree.kdtm = None
                    tree.hnsw_tree = None
                    tree.hnsw_tree_id = None
                    tree.probability = None
                    tree.n_terminals = int(tree_meta.get("n_terminals", 0))
                    tree.physical_clearance = float(tree_meta.get("physical_clearance", 0.0))
                    tree.random_seed = tree_meta.get("random_seed", None)
                    tree.characteristic_length = tree_meta.get("characteristic_length", None)
                    tree.domain_clearance = tree_meta.get("domain_clearance", None)
                    tree.clamped_root = bool(tree_meta.get("clamped_root", False))
                    tree.nonconvex_count = int(tree_meta.get("nonconvex_count", 0))
                    tree.convex = tree_meta.get("convex", None)
                    tree.max_distal_node = int(tree_meta.get("max_distal_node", n_rows) or n_rows)
                    tree.tree_scale = tree_meta.get("tree_scale", None)
                    tree.times = tree_meta.get("times", tree.times)
                    tree._analysis_only_load = True
                    if restore_progress is not None:
                        restore_progress.update(1)

            if load_progress is not None:
                load_progress.close()
            if restore_progress is not None:
                restore_progress.close()

        return forest

    @staticmethod
    def _load_archive_members(path: str, *, show_progress: bool = False):
        import numpy as np

        with zipfile.ZipFile(path, 'r') as archive:
            with archive.open('metadata.npy', 'r') as handle:
                metadata = np.load(handle, allow_pickle=True)[0]

            tree_info = archive.getinfo('trees.npy')
            gc_was_enabled = gc.isenabled()
            if gc_was_enabled:
                gc.disable()
            if show_progress:
                try:
                    with archive.open('trees.npy', 'r') as raw_handle:
                        with _ProgressReader(raw_handle, tree_info.file_size, "Loading+decoding forest trees") as handle:
                            trees_data = np.load(handle, allow_pickle=True)[0]
                finally:
                    if gc_was_enabled:
                        gc.enable()
            else:
                try:
                    with archive.open('trees.npy', 'r') as handle:
                        trees_data = np.load(handle, allow_pickle=True)[0]
                finally:
                    if gc_was_enabled:
                        gc.enable()

            if 'connections.npy' in archive.namelist():
                with archive.open('connections.npy', 'r') as handle:
                    connections_data = np.load(handle, allow_pickle=True)[0]
            else:
                connections_data = None

        return metadata, trees_data, connections_data

    def save_simulation_cache(self, path: str):
        """
        Save a simulation-oriented forest cache optimized for fast reloads.
        """
        import numpy as np

        from pathlib import Path

        path = Path(path)
        meta = {
            "format": "svv_forest_simulation_cache",
            "version": 1,
            "n_networks": int(self.n_networks),
            "n_trees_per_network": [int(v) for v in self.n_trees_per_network],
            "start_points": self.start_points,
            "directions": self.directions,
            "physical_clearance": float(self.physical_clearance),
            "compete": bool(self.compete),
            "convex": self.convex,
            "trees": [],
        }

        def _trim(tree):
            data_arr = np.asarray(tree.data)
            if data_arr.ndim != 2:
                data_arr = np.atleast_2d(data_arr)
            seg_count = int(getattr(tree, "segment_count", data_arr.shape[0]) or data_arr.shape[0])
            seg_count = max(0, min(seg_count, data_arr.shape[0]))
            return np.asarray(data_arr[:seg_count, :])

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for i in range(self.n_networks):
                network_meta = []
                for j in range(self.n_trees_per_network[i]):
                    tree = self.networks[i][j]
                    data_member = f"tree_{i}_{j}_data.npy"
                    trimmed = _trim(tree)
                    with archive.open(data_member, "w", force_zip64=True) as handle:
                        np.save(handle, trimmed, allow_pickle=False)
                    tree_meta = {
                        "data_member": data_member,
                        "segment_count": int(trimmed.shape[0]),
                        "n_terminals": int(getattr(tree, "n_terminals", 0) or 0),
                        "physical_clearance": float(getattr(tree, "physical_clearance", 0.0) or 0.0),
                        "random_seed": getattr(tree, "random_seed", None),
                        "characteristic_length": getattr(tree, "characteristic_length", None),
                        "domain_clearance": getattr(tree, "domain_clearance", None),
                        "clamped_root": bool(getattr(tree, "clamped_root", False)),
                        "nonconvex_count": int(getattr(tree, "nonconvex_count", 0) or 0),
                        "convex": getattr(tree, "convex", None),
                        "max_distal_node": int(getattr(tree, "max_distal_node", trimmed.shape[0]) or trimmed.shape[0]),
                        "tree_scale": getattr(tree, "tree_scale", None),
                        "parameters": self._serialize_tree_parameters(tree),
                    }
                    exact_connectivity = getattr(tree, "connectivity", None)
                    if exact_connectivity is not None:
                        connectivity_member = f"tree_{i}_{j}_connectivity.npy"
                        with archive.open(connectivity_member, "w", force_zip64=True) as handle:
                            np.save(
                                handle,
                                np.asarray(exact_connectivity, dtype=getattr(tree, "index_dtype", np.int64)),
                                allow_pickle=False,
                            )
                        tree_meta["connectivity_member"] = connectivity_member
                    exact_node_ids = getattr(tree, "_cascade_node_ids", None)
                    if exact_node_ids is not None:
                        node_ids_member = f"tree_{i}_{j}_node_ids.npy"
                        with archive.open(node_ids_member, "w", force_zip64=True) as handle:
                            np.save(
                                handle,
                                np.asarray(exact_node_ids, dtype=getattr(tree, "index_dtype", np.int64)),
                                allow_pickle=False,
                            )
                        tree_meta["node_ids_member"] = node_ids_member
                    network_meta.append(tree_meta)
                meta["trees"].append(network_meta)
            archive.writestr("simulation_meta.pkl", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

        return path

    @staticmethod
    def _npy_header_from_stream(handle):
        return read_npy_header_stream(handle)

    @staticmethod
    def _candidate_data_block(buffer, offset, total_trees):
        # Legacy .forest files pickle each TreeData table as a raw ndarray
        # byte string.  Simulation needs those tables and not the following
        # vessel_map object graph.
        if offset + 5 > len(buffer):
            return None
        opcode = buffer[offset]
        if opcode == 0x42:  # BINBYTES, uint32 length
            nbytes = struct.unpack("<I", buffer[offset + 1:offset + 5])[0]
            header = 5
        elif opcode == 0x8e:  # BINBYTES8, uint64 length
            if offset + 9 > len(buffer):
                return None
            nbytes = struct.unpack("<Q", buffer[offset + 1:offset + 9])[0]
            header = 9
        else:
            return None
        if nbytes <= 0:
            return None
        dtype = None
        row_bytes = None
        if (nbytes % (31 * 8)) == 0:
            dtype = "float64"
            row_bytes = 31 * 8
        elif (nbytes % (31 * 4)) == 0:
            dtype = "float32"
            row_bytes = 31 * 4
        if dtype is None:
            return None
        rows = nbytes // row_bytes
        if rows <= 0:
            return None
        # Reject tiny false positives; real tree data blocks are preceded by
        # ndarray shape/dtype construction and should be the only plausible
        # blocks before we have extracted the expected tree count.
        if rows < 1 or total_trees <= 0:
            return None
        return header, nbytes, dtype, rows

    @staticmethod
    def _last_binfloats(buffer, count=12):
        values = []
        i = 0
        while i + 9 <= len(buffer):
            if buffer[i] == 0x47:  # BINFLOAT
                values.append(struct.unpack(">d", buffer[i + 1:i + 9])[0])
                i += 9
            else:
                i += 1
        return values[-count:]

    @classmethod
    def build_simulation_cache_from_legacy(cls, source_path: str, cache_path: str, *, show_progress: bool = True):
        import numpy as np

        with zipfile.ZipFile(source_path, "r") as source:
            with source.open("metadata.npy", "r") as handle:
                metadata = np.load(handle, allow_pickle=True)[0]
            n_trees_per_network = [int(v) for v in metadata.get("n_trees_per_network", [2])]
            total_trees = sum(n_trees_per_network)
            tree_info = source.getinfo("trees.npy")

            meta = {
                "format": "svv_forest_simulation_cache",
                "version": 1,
                "n_networks": int(metadata.get("n_networks", 1)),
                "n_trees_per_network": n_trees_per_network,
                "start_points": metadata.get("start_points"),
                "directions": metadata.get("directions"),
                "physical_clearance": float(metadata.get("physical_clearance", 0.0)),
                "compete": bool(metadata.get("compete", False)),
                "convex": metadata.get("convex", None),
                "trees": [[] for _ in n_trees_per_network],
            }

            progress = None
            if show_progress:
                try:
                    from tqdm import tqdm
                    progress = tqdm(total=tree_info.file_size, desc="Extracting forest data", unit="B", unit_scale=True, unit_divisor=1024)
                except Exception:
                    progress = None

            def _read_more(src_handle, buf, min_bytes, chunk_size=8 * 1024 * 1024):
                while len(buf) < min_bytes:
                    chunk = src_handle.read(chunk_size)
                    if not chunk:
                        break
                    if progress is not None:
                        progress.update(len(chunk))
                    buf.extend(chunk)
                return buf

            def _looks_like_tree_data(buf, data_start, nbytes, dtype, rows):
                row_bytes = 31 * dtype.itemsize
                sample_rows = int(min(max(rows, 1), 4096))
                sample_bytes = int(min(nbytes, sample_rows * row_bytes))
                if len(buf) < data_start + sample_bytes or sample_bytes < row_bytes:
                    return False
                arr = np.frombuffer(buf[data_start:data_start + sample_bytes], dtype=dtype).reshape((-1, 31))
                if arr.shape[0] == 0:
                    return False

                coords = arr[:, 0:6]
                if np.isfinite(coords).mean() < 0.99:
                    return False
                if np.nanmax(np.abs(coords)) > 1.0e6:
                    return False

                lengths = arr[:, 20]
                radii = arr[:, 21]
                finite_geom = np.isfinite(lengths) & np.isfinite(radii)
                if finite_geom.mean() < 0.99:
                    return False
                if np.nanmin(lengths[finite_geom]) <= 0 or np.nanmin(radii[finite_geom]) <= 0:
                    return False
                if np.nanmax(lengths[finite_geom]) > 1.0e6 or np.nanmax(radii[finite_geom]) > 1.0e6:
                    return False

                flows = arr[:, 22]
                finite_flows = flows[np.isfinite(flows)]
                if finite_flows.size == 0 or np.nanmax(np.abs(finite_flows)) > 1.0e12:
                    return False

                conn = arr[:, 15:18]
                finite_conn = conn[np.isfinite(conn)]
                if finite_conn.size:
                    if np.nanmin(finite_conn) < 0 or np.nanmax(finite_conn) > max(rows + 1, 1):
                        return False
                    if np.nanmax(np.abs(finite_conn - np.rint(finite_conn))) > 1.0e-6:
                        return False

                distal = arr[:, 19]
                finite_distal = distal[np.isfinite(distal)]
                if finite_distal.size:
                    if np.nanmin(finite_distal) < 0 or np.nanmax(finite_distal) > max(rows + 1, 1):
                        return False
                    if np.nanmax(np.abs(finite_distal - np.rint(finite_distal))) > 1.0e-6:
                        return False

                return True

            def _stream_tree_payload(src_handle, cache, data_member, dtype, rows, initial_payload, nbytes, radius_exp, length_exp):
                row_bytes = 31 * dtype.itemsize
                terminal_count = 0
                max_distal_node = 0
                tree_scale_sum = 0.0
                first_root_flow = None
                pending = b""
                written = 0

                with cache.open(data_member, "w", force_zip64=True) as out_handle:
                    np.lib.format.write_array_header_1_0(
                        out_handle,
                        {"descr": dtype.str, "fortran_order": False, "shape": (int(rows), 31)},
                    )

                    def consume(payload):
                        nonlocal pending, terminal_count, max_distal_node, tree_scale_sum, first_root_flow, written
                        if not payload:
                            return
                        out_handle.write(payload)
                        written += len(payload)
                        data = pending + payload
                        full_len = (len(data) // row_bytes) * row_bytes
                        if full_len:
                            arr = np.frombuffer(data[:full_len], dtype=dtype).reshape((-1, 31))
                            if first_root_flow is None and arr.shape[0] > 0:
                                first_root_flow = float(arr[0, 22])
                            terminal_count += int(np.count_nonzero(np.isnan(arr[:, 15]) & np.isnan(arr[:, 16])))
                            if arr.shape[0]:
                                finite_distal = arr[:, 19][np.isfinite(arr[:, 19])]
                                if finite_distal.size:
                                    max_distal_node = max(max_distal_node, int(np.nanmax(finite_distal)))
                            tree_scale_sum += float(np.nansum((arr[:, 21] ** radius_exp) * (arr[:, 20] ** length_exp)))
                        pending = data[full_len:]

                    take = min(nbytes, len(initial_payload))
                    consume(bytes(initial_payload[:take]))
                    remaining = nbytes - take
                    while remaining > 0:
                        chunk = src_handle.read(min(8 * 1024 * 1024, remaining))
                        if not chunk:
                            raise EOFError("Unexpected end of trees.npy while streaming data block.")
                        if progress is not None:
                            progress.update(len(chunk))
                        consume(chunk)
                        remaining -= len(chunk)

                if pending:
                    raise RuntimeError("Tree data payload ended on a partial row.")
                if written != nbytes:
                    raise RuntimeError(f"Streamed {written} data bytes, expected {nbytes}.")
                return {
                    "terminal_count": int(terminal_count),
                    "max_distal_node": int(max_distal_node),
                    "tree_scale": float(np.pi * tree_scale_sum),
                    "root_flow": float(first_root_flow) if first_root_flow is not None else 1.0,
                    "trailing": bytes(initial_payload[take:]),
                }

            with source.open("trees.npy", "r") as raw_trees:
                cls._npy_header_from_stream(raw_trees)
                buffer = bytearray()
                scan = 0
                tree_index = 0
                with zipfile.ZipFile(cache_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as cache:
                    while tree_index < total_trees:
                        buffer = _read_more(raw_trees, buffer, scan + 16)
                        if scan >= len(buffer):
                            raise EOFError(f"Could not find all tree data blocks; extracted {tree_index}/{total_trees}.")
                        found = False
                        while scan < len(buffer):
                            candidate = cls._candidate_data_block(buffer, scan, total_trees)
                            if candidate is None:
                                scan += 1
                                continue
                            header, nbytes, dtype_name, rows = candidate

                            net_id = 0
                            local_id = tree_index
                            running = 0
                            for idx, count in enumerate(n_trees_per_network):
                                if tree_index < running + count:
                                    net_id = idx
                                    local_id = tree_index - running
                                    break
                                running += count

                            data_member = f"tree_{net_id}_{local_id}_data.npy"
                            dtype = np.dtype(dtype_name)
                            data_start = scan + header
                            sample_bytes = min(nbytes, min(max(rows, 1), 4096) * 31 * dtype.itemsize)
                            buffer = _read_more(raw_trees, buffer, data_start + sample_bytes)
                            if not _looks_like_tree_data(buffer, data_start, nbytes, dtype, rows):
                                scan += 1
                                continue
                            params_floats = cls._last_binfloats(buffer[max(0, scan - 4096):scan], count=12)
                            param_tail = params_floats[-9:] if len(params_floats) >= 9 else []
                            if len(param_tail) == 9:
                                kin_visc, density, terminal_flow, root_flow, terminal_pressure, root_pressure, murray, radius_exp, length_exp = param_tail
                            else:
                                kin_visc, density = 0.011320754716981131, 1.06
                                terminal_flow = None
                                root_flow = None
                                terminal_pressure, root_pressure = 5999.51, 12665.6
                                murray, radius_exp, length_exp = 3.0, 2.0, 1.0

                            stream_stats = _stream_tree_payload(
                                raw_trees,
                                cache,
                                data_member,
                                dtype,
                                rows,
                                memoryview(buffer)[data_start:],
                                nbytes,
                                radius_exp,
                                length_exp,
                            )
                            if terminal_flow is None:
                                terminal_flow = stream_stats["root_flow"] / max(1, int(stream_stats["terminal_count"]))
                            if root_flow is None:
                                root_flow = stream_stats["root_flow"]

                            meta["trees"][net_id].append({
                                "data_member": data_member,
                                "segment_count": int(rows),
                                "n_terminals": int(stream_stats["terminal_count"]),
                                "physical_clearance": float(metadata.get("physical_clearance", 0.0)),
                                "random_seed": None,
                                "characteristic_length": None,
                                "domain_clearance": 0.0,
                                "clamped_root": True,
                                "nonconvex_count": 0,
                                "convex": metadata.get("convex", None),
                                "max_distal_node": int(stream_stats["max_distal_node"]),
                                "tree_scale": float(stream_stats["tree_scale"]),
                                "parameters": {
                                    "kinematic_viscosity": float(kin_visc),
                                    "fluid_density": float(density),
                                    "terminal_flow": float(terminal_flow),
                                    "root_flow": float(root_flow),
                                    "terminal_pressure": float(terminal_pressure),
                                    "root_pressure": float(root_pressure),
                                    "murray_exponent": float(murray),
                                    "radius_exponent": float(radius_exp),
                                    "length_exponent": float(length_exp),
                                    "max_nonconvex_count": 100,
                                    "unit_system": {
                                        "length": "cm",
                                        "time": "s",
                                        "mass": "g",
                                        "pressure": "g/(cm * s^2)",
                                    },
                                },
                            })

                            tree_index += 1
                            buffer = bytearray(stream_stats["trailing"])
                            scan = 0
                            found = True
                            break
                    cache.writestr("simulation_meta.pkl", pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))

            if progress is not None:
                progress.close()
            if tree_index != total_trees:
                raise RuntimeError(f"Extracted {tree_index} tree data blocks, expected {total_trees}.")

        return cache_path

    @classmethod
    def load(cls, path: str, *, mode: str = "growth", data_dtype=None, index_dtype=None):
        """
        Load a Forest from a .forest file.

        The loaded forest will NOT have a domain set. You must call
        :meth:`set_domain` after loading to enable domain-dependent
        operations like collision detection and further vessel generation.

        Parameters
        ----------
        path : str
            Path to a .forest file.

        Returns
        -------
        Forest
            Loaded forest instance.

        Notes
        -----
        If the forest was saved with connections (after calling :meth:`connect`),
        the connection results (vessels, assignments, connected_network) will be
        restored. However, to re-solve connections or generate new vessels,
        you must first call :meth:`set_domain`.  During loading, per-tree
        preallocation buffers and spatial indices (KD-tree / USearch) are
        rebuilt from the stored vessel table so growth can continue.  The
        sampling probability distribution is domain-dependent and therefore
        re-seeded when :meth:`set_domain` is called.

        Examples
        --------
        >>> forest = Forest.load("my_forest.forest")
        >>> forest.set_domain(domain)  # Required for domain operations
        >>> forest.show()
        """
        import numpy as np
        from svv.tree.data.data import TreeData, TreeParameters, TreeMap
        from svv.tree.data.units import UnitSystem
        from svv.tree.utils.TreeManager import KDTreeManager, USearchTree

        mode = cls._resolve_load_mode(mode)
        if cls._is_simulation_cache(path):
            if mode != "simulation":
                raise ValueError("Simulation cache archives can only be loaded with mode='simulation'.")
            return cls._load_simulation_cache(path, data_dtype=data_dtype, index_dtype=index_dtype)

        show_progress = mode == "simulation"
        metadata, trees_data, connections_data = cls._load_archive_members(path, show_progress=show_progress)

        # Check version
        version = metadata.get('version', 1)
        if version > 1:
            raise ValueError(f"Unsupported .forest file version: {version}")

        # Estimate a reasonable preallocation size from stored tree data to
        # avoid allocating extremely large default arrays when loading.  Older
        # .forest files may include full preallocation buffers; prefer the
        # recorded segment_count when present.
        max_rows = 1
        for i in range(metadata['n_networks']):
            for j in range(metadata['n_trees_per_network'][i]):
                td = trees_data[i][j]
                data_arr = td['data']
                seg_count = None
                try:
                    seg_count = int(td.get('metadata', {}).get('segment_count', 0))
                except Exception:
                    seg_count = None
                if seg_count is not None and seg_count > max_rows:
                    max_rows = seg_count
                elif hasattr(data_arr, 'shape') and data_arr.shape[0] > max_rows:
                    max_rows = int(data_arr.shape[0])
        if mode == "simulation":
            # The populated arrays replace these constructor buffers below.
            # Starting at one row avoids a second multi-gigabyte allocation.
            preallocation_step = 1
        else:
            preallocation_step = max(max_rows * 2, 1)

        # Reconstruct start_points and directions
        start_points = []
        for network_pts in metadata.get('start_points', []):
            network_list = []
            for pt in network_pts:
                if pt is not None:
                    network_list.append(numpy.array(pt))
                else:
                    network_list.append(None)
            start_points.append(network_list)

        directions = []
        for network_dirs in metadata.get('directions', []):
            network_list = []
            for d in network_dirs:
                if d is not None:
                    network_list.append(numpy.array(d))
                else:
                    network_list.append(None)
            directions.append(network_list)

        # Create forest with metadata
        forest = cls(
            n_networks=metadata['n_networks'],
            n_trees_per_network=metadata['n_trees_per_network'],
            start_points=start_points if start_points else None,
            directions=directions if directions else None,
            physical_clearance=metadata.get('physical_clearance', 0.0),
            compete=metadata.get('compete', False),
            preallocation_step=preallocation_step,
        )
        forest.convex = metadata.get('convex', None)

        tree_progress = None
        if show_progress:
            try:
                from tqdm import tqdm
                total_trees = sum(int(v) for v in forest.n_trees_per_network)
                tree_progress = tqdm(total=total_trees, desc="Restoring forest trees", unit="tree")
            except Exception:
                tree_progress = None

        # Restore each tree's data
        for i in range(forest.n_networks):
            for j in range(forest.n_trees_per_network[i]):
                tree_dict = trees_data[i][j]
                tree = forest.networks[i][j]

                # Restore unit system from base units only; pressure and other
                # derived quantities are always auto-derived from these.
                us_dict = tree_dict['parameters'].get('unit_system', {})
                length_unit = us_dict.get('length', 'cm')
                time_unit = us_dict.get('time', 's')
                mass_unit = us_dict.get('mass', 'g')
                unit_system = UnitSystem(
                    length=length_unit,
                    time=time_unit,
                    mass=mass_unit,
                )
                tree.parameters.set_unit_system(unit_system)

                # Restore parameters
                params = tree_dict['parameters']
                tree.parameters.kinematic_viscosity = params['kinematic_viscosity']
                tree.parameters.fluid_density = params['fluid_density']
                tree.parameters.terminal_flow = params['terminal_flow']
                tree.parameters.root_flow = params['root_flow']
                tree.parameters.terminal_pressure = params['terminal_pressure']
                tree.parameters.root_pressure = params['root_pressure']
                tree.parameters.murray_exponent = params['murray_exponent']
                tree.parameters.radius_exponent = params['radius_exponent']
                tree.parameters.length_exponent = params['length_exponent']
                tree.parameters.max_nonconvex_count = params['max_nonconvex_count']

                # Restore data
                data_array = np.asarray(tree_dict['data'])
                if data_array.ndim != 2:
                    data_array = np.atleast_2d(data_array)
                if data_dtype is not None:
                    requested_data_dtype = np.dtype(data_dtype)
                    if requested_data_dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
                        raise ValueError("data_dtype must be float32 or float64.")
                    data_array = np.asarray(data_array, dtype=requested_data_dtype)
                requested_index_dtype = np.dtype(np.int64 if index_dtype is None else index_dtype)
                if requested_index_dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
                    raise ValueError("index_dtype must be int32 or int64.")
                tree.data = TreeData.from_array(data_array)
                tree.data_dtype = np.dtype(data_array.dtype)
                tree.index_dtype = requested_index_dtype

                # Determine the number of populated vessel rows.  Prefer the
                # saved segment_count, but fall back to trimming trailing NaNs
                # for legacy files that stored preallocated buffers.
                n_rows = 0
                meta_seg = tree_dict.get('metadata', {}).get('segment_count', None)
                try:
                    meta_seg = int(meta_seg)
                except (TypeError, ValueError):
                    meta_seg = None
                if meta_seg is not None and 0 < meta_seg <= data_array.shape[0]:
                    n_rows = meta_seg
                else:
                    valid = ~np.all(np.isnan(data_array), axis=1)
                    if valid.any():
                        n_rows = int(np.where(valid)[0][-1] + 1)
                if n_rows <= 0:
                    n_rows = 1

                # Trim the in-memory vessel table to populated rows to avoid
                # keeping legacy preallocation buffers alive after load.
                if n_rows < data_array.shape[0]:
                    data_array = data_array[:n_rows, :]
                    tree.data = TreeData.from_array(data_array)

                # Analysis-only loads alias the populated vessel table rather
                # than retaining a second growth buffer. Growth-mode loads keep
                # the expandable buffers and indices needed to add vessels.
                if mode == "simulation":
                    tree.preallocation_step = n_rows
                    tree.preallocate = tree.data
                    tree.preallocate_midpoints = np.empty((0, 3), dtype=tree.data_dtype)
                    tree.midpoints = tree.preallocate_midpoints
                elif n_rows > tree.preallocate.shape[0]:
                    new_size = max(n_rows * 2, tree.preallocate.shape[0])
                    tree.preallocation_step = int(new_size)
                    tree.preallocate = TreeData((new_size, tree.data.shape[1]))
                    tree.preallocate_midpoints = np.zeros((new_size, 3))

                if mode != "simulation":
                    tree.preallocate[:n_rows, :] = tree.data[:n_rows, :]
                    midpoints = (tree.data[:n_rows, 0:3] + tree.data[:n_rows, 3:6]) / 2
                    tree.preallocate_midpoints[:n_rows, :] = midpoints
                    tree.midpoints = tree.preallocate_midpoints[:n_rows, :]
                else:
                    midpoints = np.empty((0, 3), dtype=tree.data_dtype)

                # Rebuild connectivity/search structures derived from data.
                tree.connectivity = (
                    None
                    if mode == "simulation"
                    else np.nan_to_num(tree.data[:n_rows, 15:18], nan=-1.0).astype(requested_index_dtype)
                )
                if mode == "simulation":
                    tree.kdtm = None
                    tree.hnsw_tree = None
                    tree.hnsw_tree_id = None
                else:
                    tree.kdtm = KDTreeManager(midpoints)
                    tree.hnsw_tree = USearchTree(midpoints.astype(np.float32))
                    tree.hnsw_tree_id = id(tree.hnsw_tree)
                distal_nodes = tree.data[:n_rows, 19]
                if n_rows and not np.all(np.isnan(distal_nodes)):
                    tree.max_distal_node = int(np.nanmax(distal_nodes))
                else:
                    tree.max_distal_node = n_rows
                tree.tree_scale = float(
                    np.pi * np.nansum(
                        (tree.data[:n_rows, 21] ** tree.parameters.radius_exponent)
                        * (tree.data[:n_rows, 20] ** tree.parameters.length_exponent)
                    )
                )

                # Restore vessel map
                if mode == "simulation":
                    tree.vessel_map = TreeMap()
                else:
                    tree.vessel_map = TreeMap(tree_dict['vessel_map'])

                # Restore metadata
                meta = tree_dict['metadata']
                tree.n_terminals = meta.get('n_terminals', 0)
                tree.physical_clearance = meta.get('physical_clearance', 0.0)
                tree.random_seed = meta.get('random_seed', None)
                tree.characteristic_length = meta.get('characteristic_length', None)
                tree.clamped_root = meta.get('clamped_root', False)
                tree.nonconvex_count = meta.get('nonconvex_count', 0)
                tree.convex = meta.get('convex', None)
                # Segment count must align with the loaded data length.
                tree.segment_count = n_rows
                tree._analysis_only_load = mode == "simulation"
                tree.domain_clearance = meta.get('domain_clearance', 0.0)
                # Optional persisted values (older files may omit them).
                if meta.get('max_distal_node') is not None:
                    tree.max_distal_node = meta['max_distal_node']
                if meta.get('tree_scale') is not None:
                    tree.tree_scale = meta['tree_scale']

                # Restore timing if present
                if 'times' in tree_dict:
                    tree.times = tree_dict['times']
                if tree_progress is not None:
                    tree_progress.update(1)

        if tree_progress is not None:
            tree_progress.close()

        # Restore connections if present
        if mode != "simulation" and connections_data is not None:
            from svv.forest.connect.forest_connection import ForestConnection

            forest.connections = ForestConnection(forest)
            forest.connections.tree_connections = []

            for tc_data in connections_data['tree_connections']:
                # Create a minimal TreeConnection-like object to hold the data
                tc = _LoadedTreeConnection(
                    forest=forest,
                    network_id=tc_data['network_id'],
                    assignments=tc_data['assignments'],
                    vessels=tc_data['vessels'],
                    lengths=tc_data['lengths'],
                    curve_type=tc_data['curve_type'],
                    connected_network_data=tc_data['connected_network'],
                )
                forest.connections.tree_connections.append(tc)

        return forest


class _LoadedTreeConnection:
    """
    Minimal container for loaded TreeConnection data.

    This class holds the results of a previously solved tree connection
    without requiring access to the original domain or solving infrastructure.
    """

    def __init__(self, forest, network_id, assignments, vessels, lengths,
                 curve_type, connected_network_data):
        from svv.tree.data.data import TreeData
        from cascade.vessels.generation.svv_adapter import Tree as _adapter_tree

        self.forest = forest
        self.network_id = network_id
        self.assignments = assignments
        self.vessels = vessels
        self.lengths = lengths
        self.curve_type = curve_type
        self.other_vessels = []
        self.meshes = []
        self.plotting_vessels = None
        self.connections = []

        # Reconstruct connected_network as Tree objects with data
        self.connected_network = []
        for i, data_array in enumerate(connected_network_data):
            # Use a small preallocation sized to the stored vessel table.
            # The loaded connection trees are lightweight containers for
            # geometry export and should not allocate the multi‑million row
            # growth buffers that `Tree()` defaults to.
            n_rows = int(getattr(data_array, "shape", (1,))[0]) or 1
            tree = _adapter_tree(preallocation_step=max(n_rows * 2, 1))
            tree.parameters = forest.networks[network_id][i].parameters
            tree.data = TreeData.from_array(data_array)
            tree.domain = None  # Will be set when forest.set_domain() is called
            self.connected_network.append(tree)

        # Build network reference (matches TreeConnection structure)
        self.network = []
        for i in range(forest.n_trees_per_network[network_id]):
            n_rows = int(getattr(forest.networks[network_id][i].data, "shape", (1,))[0]) or 1
            tree = _adapter_tree(preallocation_step=max(n_rows * 2, 1))
            tree.parameters = forest.networks[network_id][i].parameters
            tree.data = forest.networks[network_id][i].data
            tree.domain = None
            self.network.append(tree)

    def export_solid(self, cap_resolution=40, extrude_roots=False):
        """Export connected network solids for this loaded connection.

        The original :class:`~svv.forest.connect.tree_connection.TreeConnection`
        implementation provides the meshing logic.  Loaded connections already
        contain the solved assignments, vessels, and connected_network, so we
        delegate to that method after populating a lightweight stand‑in
        instance.  This avoids duplicating a large amount of geometry code
        while keeping legacy `.forest` files functional.

        Parameters
        ----------
        cap_resolution : int, optional
            Resolution for end‑cap triangulation passed through to the
            underlying export routine.
        extrude_roots : bool, optional
            When True, extend root vessels slightly outside the domain before
            meshing (requires the Forest domain to be set).
        """
        from svv.forest.connect.tree_connection import TreeConnection

        # Create an uninitialized TreeConnection and graft the loaded state
        # onto it so we can reuse TreeConnection.export_solid.
        tmp = TreeConnection.__new__(TreeConnection)
        tmp.forest = self.forest
        tmp.network_id = self.network_id
        tmp.assignments = self.assignments
        tmp.ctrlpts_functions = None
        tmp.connections = self.connections
        tmp.vessels = self.vessels
        tmp.lengths = self.lengths
        tmp.other_vessels = self.other_vessels
        tmp.meshes = self.meshes
        tmp.plotting_vessels = self.plotting_vessels
        tmp.network = self.network
        tmp.curve_type = self.curve_type
        tmp.connected_network = self.connected_network

        return TreeConnection.export_solid(
            tmp,
            cap_resolution=cap_resolution,
            extrude_roots=extrude_roots,
        )
