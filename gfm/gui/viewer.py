"""Standalone, memory-isolated PyVista result viewer.

The viewer deliberately does not embed VTK in the main Qt application. That
keeps graphics allocations out of CASCADE Studio and is more reliable on WSLg.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyvista as pv


COLORMAPS = ("viridis", "magma", "plasma", "inferno", "coolwarm", "jet", "gray")


class ResultViewer:
    def __init__(
        self,
        manifest_path: str | Path,
        *,
        colormap: str = "viridis",
        max_vessels: int = 1000,
        vessel_scalar: str = "flow_ul_min",
        tissue_scalar: str = "local_concentration",
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.vessels = self._read_output("vessels_vtp")
        self.tissue = self._read_output("oxygen_points_vtp")
        self.domain = self._read_output("domain_boundary_vtp")
        if self.vessels is None and self.tissue is None and self.domain is None:
            raise ValueError(
                "No VTK outputs were found. Enable ParaView VTK files and rerun the simulation."
            )
        self._add_normalized_fields()
        self.vessel_scalars = self._array_names(self.vessels)
        self.tissue_scalars = self._array_names(self.tissue)
        self.vessel_scalar_index = self._index(self.vessel_scalars, vessel_scalar)
        self.tissue_scalar_index = self._index(self.tissue_scalars, tissue_scalar)
        self.cmap_index = self._index(list(COLORMAPS), colormap)
        self.max_vessels = max(1, int(max_vessels))
        self.slice_axis = 2
        self.slice_width = 0.02
        self.slice_position = 0.0
        self.slice_enabled = False
        self.plotter = pv.Plotter(
            title=f"CASCADE result · {self.manifest_path.parent.name}"
        )
        self.plotter.set_background("#f4f6f7")
        self.vessel_actor = None
        self.tissue_actor = None
        self.domain_actor = None

    def show(self) -> None:
        self._add_meshes()
        self._add_controls()
        self.plotter.add_axes()
        self.plotter.reset_camera()
        self.plotter.show()

    def _read_output(self, key: str):
        value = self.manifest.get("outputs", {}).get(key)
        if not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        try:
            return pv.read(str(path)) if path.exists() else None
        except Exception:
            return None

    def _add_normalized_fields(self) -> None:
        if self.vessels is None:
            return
        settings = self.manifest.get("settings", {})
        qin = float(settings.get("simulation", {}).get("qin_target_ul_min", 0.0))
        if qin > 0.0 and "flow_ul_min" in self.vessels.point_data:
            self.vessels.point_data["flow / total inlet flow"] = (
                np.asarray(self.vessels.point_data["flow_ul_min"], dtype=float) / qin
            )
        oxygen = settings.get("settings", {}).get("oxygen", {})
        inlet = oxygen.get(
            "conc_max_for_normalization", oxygen.get("CONC_MAX_FOR_NORMALIZATION", 0.0)
        )
        try:
            inlet = float(inlet)
        except Exception:
            inlet = 0.0
        if inlet > 0.0 and "concentration" in self.vessels.point_data:
            self.vessels.point_data["oxygen / inlet oxygen"] = (
                np.asarray(self.vessels.point_data["concentration"], dtype=float)
                / inlet
            )

    def _add_meshes(self) -> None:
        cmap = COLORMAPS[self.cmap_index]
        if self.domain is not None:
            self.domain_actor = self.plotter.add_mesh(
                self.domain, color="#9fb4bc", opacity=0.16, name="domain"
            )
        if self.vessels is not None:
            subset = self._vessel_subset(self.max_vessels)
            scalar = self._scalar(self.vessel_scalars, self.vessel_scalar_index)
            self.vessel_actor = self.plotter.add_mesh(
                subset,
                scalars=scalar,
                cmap=cmap,
                opacity=1.0,
                line_width=3,
                render_lines_as_tubes=False,
                name="vessels",
                scalar_bar_args={"title": scalar or "Vessels"},
            )
        if self.tissue is not None:
            scalar = self._scalar(self.tissue_scalars, self.tissue_scalar_index)
            self.tissue_actor = self.plotter.add_mesh(
                self.tissue,
                scalars=scalar,
                cmap=cmap,
                opacity=0.45,
                point_size=5,
                render_points_as_spheres=False,
                name="tissue",
                scalar_bar_args={"title": scalar or "Tissue"},
            )
        self._update_status()

    def _add_controls(self) -> None:
        y = 12
        for label, actor in (
            ("Vessels", self.vessel_actor),
            ("Tissue", self.tissue_actor),
            ("Domain", self.domain_actor),
        ):
            if actor is None:
                continue
            self.plotter.add_checkbox_button_widget(
                lambda state, current=actor: current.SetVisibility(bool(state)),
                value=True,
                position=(12, y),
                size=24,
                color_on="#2b7f90",
                color_off="#c8d1d5",
            )
            self.plotter.add_text(
                label, position=(44, y + 2), font_size=9, color="#243c47"
            )
            y += 34

        if self.vessel_actor is not None:
            self.plotter.add_slider_widget(
                lambda value: self.vessel_actor.GetProperty().SetOpacity(float(value)),
                rng=(0.0, 1.0),
                value=1.0,
                title="Vessel opacity",
                pointa=(0.03, 0.90),
                pointb=(0.28, 0.90),
                interaction_event="always",
            )
            total = max(int(self.vessels.n_cells), 1)
            if total > 1:
                self.plotter.add_slider_widget(
                    self._set_vessel_count,
                    rng=(1, total),
                    value=min(self.max_vessels, total),
                    title="Vessels shown",
                    pointa=(0.36, 0.90),
                    pointb=(0.64, 0.90),
                    interaction_event="end",
                )
        if self.tissue_actor is not None:
            self.plotter.add_slider_widget(
                lambda value: self.tissue_actor.GetProperty().SetOpacity(float(value)),
                rng=(0.0, 1.0),
                value=0.45,
                title="Tissue opacity",
                pointa=(0.72, 0.90),
                pointb=(0.97, 0.90),
                interaction_event="always",
            )
            bounds = self.tissue.bounds
            self.slice_position = 0.5 * (bounds[4] + bounds[5])
            self.slice_width = max((bounds[5] - bounds[4]) / 50.0, 1e-6)
            self.plotter.add_slider_widget(
                self._set_slice_position,
                rng=(bounds[4], bounds[5]),
                value=self.slice_position,
                title="Tissue slice Z",
                pointa=(0.36, 0.82),
                pointb=(0.64, 0.82),
                interaction_event="always",
            )

        self.plotter.add_key_event("v", self._cycle_vessel_scalar)
        self.plotter.add_key_event("t", self._cycle_tissue_scalar)
        self.plotter.add_key_event("c", self._cycle_colormap)
        self.plotter.add_key_event("s", self._toggle_slice)
        self.plotter.add_key_event("x", lambda: self._set_slice_axis(0))
        self.plotter.add_key_event("y", lambda: self._set_slice_axis(1))
        self.plotter.add_key_event("z", lambda: self._set_slice_axis(2))
        self.plotter.add_text(
            "Keys: V vessel field · T tissue field · C colormap · S slice on/off · X/Y/Z slice axis",
            position="lower_left",
            font_size=9,
            color="#294653",
            name="help",
        )

    def _vessel_subset(self, count: int):
        count = min(max(int(count), 1), int(self.vessels.n_cells))
        if count >= self.vessels.n_cells:
            return self.vessels
        return self.vessels.extract_cells(
            np.arange(count, dtype=np.int64)
        ).extract_surface()

    def _set_vessel_count(self, value: float) -> None:
        count = max(1, int(round(value)))
        self.max_vessels = count
        self.vessel_actor.mapper.dataset = self._vessel_subset(count)
        self._apply_scalar(
            self.vessel_actor, self.vessel_scalars, self.vessel_scalar_index
        )
        self._update_status()
        self.plotter.render()

    def _set_slice_position(self, value: float) -> None:
        self.slice_position = float(value)
        if self.slice_enabled:
            self._update_tissue_slice()

    def _toggle_slice(self) -> None:
        self.slice_enabled = not self.slice_enabled
        self._update_tissue_slice()

    def _set_slice_axis(self, axis: int) -> None:
        self.slice_axis = int(axis)
        if self.tissue is not None:
            bounds = self.tissue.bounds
            lo, hi = bounds[2 * axis], bounds[2 * axis + 1]
            self.slice_position = 0.5 * (lo + hi)
            self.slice_width = max((hi - lo) / 50.0, 1e-6)
        if self.slice_enabled:
            self._update_tissue_slice()

    def _update_tissue_slice(self) -> None:
        if self.tissue_actor is None:
            return
        mesh = self.tissue
        if self.slice_enabled and mesh.n_points:
            distance = np.abs(
                np.asarray(mesh.points)[:, self.slice_axis] - self.slice_position
            )
            ids = np.flatnonzero(distance <= 0.5 * self.slice_width)
            mesh = mesh.extract_points(ids, adjacent_cells=False)
        self.tissue_actor.mapper.dataset = mesh
        self._apply_scalar(
            self.tissue_actor, self.tissue_scalars, self.tissue_scalar_index
        )
        self._update_status()
        self.plotter.render()

    def _cycle_vessel_scalar(self) -> None:
        if self.vessel_actor is None or not self.vessel_scalars:
            return
        self.vessel_scalar_index = (self.vessel_scalar_index + 1) % len(
            self.vessel_scalars
        )
        self._apply_scalar(
            self.vessel_actor, self.vessel_scalars, self.vessel_scalar_index
        )
        self._update_status()
        self.plotter.render()

    def _cycle_tissue_scalar(self) -> None:
        if self.tissue_actor is None or not self.tissue_scalars:
            return
        self.tissue_scalar_index = (self.tissue_scalar_index + 1) % len(
            self.tissue_scalars
        )
        self._apply_scalar(
            self.tissue_actor, self.tissue_scalars, self.tissue_scalar_index
        )
        self._update_status()
        self.plotter.render()

    def _cycle_colormap(self) -> None:
        self.cmap_index = (self.cmap_index + 1) % len(COLORMAPS)
        for actor in (self.vessel_actor, self.tissue_actor):
            if actor is not None and actor.mapper.lookup_table is not None:
                actor.mapper.lookup_table.apply_cmap(
                    COLORMAPS[self.cmap_index], n_values=256
                )
        self._update_status()
        self.plotter.render()

    @staticmethod
    def _apply_scalar(actor, names: list[str], index: int) -> None:
        name = ResultViewer._scalar(names, index)
        if not name:
            return
        actor.mapper.array_name = name
        try:
            actor.mapper.scalar_range = actor.mapper.dataset.get_data_range(name)
        except Exception:
            pass

    def _update_status(self) -> None:
        vessel = self._scalar(self.vessel_scalars, self.vessel_scalar_index) or "none"
        tissue = self._scalar(self.tissue_scalars, self.tissue_scalar_index) or "none"
        slice_text = (
            "off"
            if not self.slice_enabled
            else f"{'xyz'[self.slice_axis]}={self.slice_position:g} cm"
        )
        shown = (
            min(self.max_vessels, int(self.vessels.n_cells))
            if self.vessels is not None
            else 0
        )
        self.plotter.add_text(
            f"Vessels: {vessel} ({shown:,} shown)\nTissue: {tissue} · colormap: {COLORMAPS[self.cmap_index]} · slice: {slice_text}",
            position="upper_right",
            font_size=9,
            color="#294653",
            name="status",
        )

    @staticmethod
    def _array_names(mesh) -> list[str]:
        if mesh is None:
            return []
        names = []
        for name in [*mesh.point_data.keys(), *mesh.cell_data.keys()]:
            try:
                values = np.asarray(mesh[name])
                if np.issubdtype(values.dtype, np.number) and values.ndim == 1:
                    names.append(name)
            except Exception:
                pass
        return list(dict.fromkeys(names))

    @staticmethod
    def _index(values: list[str], preferred: str) -> int:
        try:
            return values.index(preferred)
        except ValueError:
            return 0

    @staticmethod
    def _scalar(values: list[str], index: int) -> str | None:
        return values[index] if values else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open a CASCADE result manifest in the external 3D viewer."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--colormap", choices=COLORMAPS, default="viridis")
    parser.add_argument("--max-vessels", type=int, default=1000)
    parser.add_argument("--vessel-scalar", default="flow_ul_min")
    parser.add_argument("--tissue-scalar", default="local_concentration")
    args = parser.parse_args(argv)
    viewer = ResultViewer(
        args.manifest,
        colormap=args.colormap,
        max_vessels=args.max_vessels,
        vessel_scalar=args.vessel_scalar,
        tissue_scalar=args.tissue_scalar,
    )
    viewer.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
