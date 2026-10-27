"""파이프라인 산출 그림 두 장.

figure1 — 가중치 지도: 기동성·은닉 지형 가중과 초기 belief 셀 질량.
figure2 — step 별 비행체 이동: 셀 중심 경유 본 항적 (항적 규약) + 실제 발진 지점.

챕터 그림(`cpp_search.figures`)과 같은 폰트·저장 규칙을 쓴다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cpp_search.figures import _configure_korean_font, _text
from cpp_search.models.terrain import TerrainField
from cpp_search.solvers.milp.stone_spx import StoneGridInstance, StoneSPXRoutePlan

_configure_korean_font()

_VEHICLE_COLORS = (
    "#0e7490", "#b45309", "#15803d", "#7c3aed", "#be123c", "#334155",
    "#0891b2", "#a16207", "#166534", "#6d28d9", "#9f1239", "#1e293b",
)


def weight_figure(
    instance: StoneGridInstance, terrain: TerrainField, path: Path
) -> Path:
    """figure1 — 입력 가중치가 모형에 어떻게 들어갔는가."""

    radius = instance.mission.search_radius_m
    axis = np.linspace(-radius, radius, 160)
    xx, yy = np.meshgrid(axis, axis)
    inside = xx**2 + yy**2 <= radius**2

    def field(fn) -> np.ndarray:
        values = np.full(xx.shape, np.nan)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                if inside[i, j]:
                    values[i, j] = fn(float(xx[i, j]), float(yy[i, j]))
        return values

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
    panels = (
        (axes[0], field(terrain.mobility_weight),
         _text("지형 가중치 (기동성)", "terrain mobility weight"), "viridis"),
        (axes[1], field(terrain.concealment_weight),
         _text("은닉 가중치", "concealment weight"), "magma"),
    )
    for ax, values, title, cmap in panels:
        image = ax.imshow(
            values,
            extent=(-radius, radius, -radius, radius),
            origin="lower",
            cmap=cmap,
        )
        figure.colorbar(image, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")

    # 초기 belief 셀 질량 — 계획이 실제로 본 값 (여러 표적이면 평균).
    mass = np.asarray(instance.problem.initial_mass, dtype=float)
    if mass.ndim > 1:
        mass = mass.mean(axis=0)
    cells = min(len(mass), instance.cell_count)
    centers = np.array(
        [[instance.grid.center_of(i).x, instance.grid.center_of(i).y]
         for i in range(cells)]
    )
    scatter = axes[2].scatter(
        centers[:, 0], centers[:, 1], c=mass[:cells], cmap="cividis",
        s=_marker_area(axes[2], instance, radius), marker="s",
    )
    figure.colorbar(scatter, ax=axes[2], shrink=0.85)
    axes[2].set_title(
        _text("초기 belief 셀 질량 (이송 확산 반영)", "initial belief cell mass"),
        fontsize=10,
    )
    for ax in axes:
        circle = plt.Circle((0, 0), radius, fill=False, color="white", ls="--", lw=1.0)
        ax.add_patch(circle)
        ax.set_xlim(-radius * 1.04, radius * 1.04)
        ax.set_ylim(-radius * 1.04, radius * 1.04)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    figure.suptitle(
        _text("figure 1 — 입력 가중치와 계획 belief", "inputs as the planner saw them"),
        fontsize=11,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def track_figure(
    instance: StoneGridInstance, plan: StoneSPXRoutePlan, path: Path
) -> Path:
    """figure2 — step 별 1~N 비행체 이동 (본 표는 셀 중심 경유)."""

    radius = instance.mission.search_radius_m
    figure, ax = plt.subplots(figsize=(7.6, 7.6), constrained_layout=True)

    _draw_grid(ax, instance, radius)
    ax.add_patch(
        plt.Circle((0, 0), radius, fill=False, color="#b45309", ls="--", lw=1.4)
    )

    for index, cell_path in enumerate(plan.assignments):
        color = _VEHICLE_COLORS[index % len(_VEHICLE_COLORS)]
        start = instance.initial_positions[index]
        points = [(start.x, start.y)] + [
            (instance.grid.center_of(cell).x, instance.grid.center_of(cell).y)
            for cell in cell_path
        ]
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=1.6, alpha=0.9,
                label=f"UAV {index + 1}")
        ax.plot(start.x, start.y, marker="x", color=color, ms=9, mew=2.2)
        step = max(1, len(cell_path) // 6)
        for slice_index in range(0, len(cell_path), step):
            ax.annotate(
                str(slice_index + 1),
                (xs[slice_index + 1], ys[slice_index + 1]),
                fontsize=6.5, color=color,
                xytext=(3, 3), textcoords="offset points",
            )

    ax.set_xlim(-radius * 1.06, radius * 1.06)
    ax.set_ylim(-radius * 1.06, radius * 1.06)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=8, loc="upper right")
    shape = instance.grid.grid_shape
    ax.set_title(
        _text(
            f"figure 2 — step 별 비행체 이동 (셀 중심 경유, ×=발진 지점)\n"
            f"{shape[0]}×{shape[1]} 격자 · 슬라이스 {instance.time_slice_count}개 "
            f"({instance.slice_s:.0f} s)",
            "per-slice vehicle movement over the planning grid",
        ),
        fontsize=10,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _draw_grid(ax, instance: StoneGridInstance, radius: float) -> None:
    kind = str(instance.config.grid_kind)
    if kind == "square":
        width, height = instance.grid.grid_shape
        xs = np.linspace(-radius, radius, width + 1)
        ys = np.linspace(-radius, radius, height + 1)
        for x in xs:
            ax.plot([x, x], [ys[0], ys[-1]], color="0.75", lw=0.5, zorder=0)
        for y in ys:
            ax.plot([xs[0], xs[-1]], [y, y], color="0.75", lw=0.5, zorder=0)
    else:  # operational-polar
        step = instance.config.radial_step_m
        ring = step
        while ring <= radius + 1e-6:
            ax.add_patch(
                plt.Circle((0, 0), ring, fill=False, color="0.8", lw=0.5, zorder=0)
            )
            ring += step
        bins = int(instance.config.angular_bin_count)
        for k in range(bins):
            angle = 2.0 * np.pi * k / bins
            ax.plot(
                [0, radius * np.cos(angle)], [0, radius * np.sin(angle)],
                color="0.85", lw=0.4, zorder=0,
            )


def _marker_area(ax, instance: StoneGridInstance, radius: float) -> float:
    """셀 하나가 차지하는 대략의 마커 크기 (pt^2)."""

    cells_across = max(instance.grid.grid_shape)
    figure = ax.get_figure()
    axis_pt = ax.get_position().width * figure.get_size_inches()[0] * 72.0
    marker_pt = max(2.0, 0.85 * axis_pt / max(cells_across, 1))
    return marker_pt**2
