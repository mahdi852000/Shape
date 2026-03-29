import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Rectangle


class ShapeApp:
    MAX_THICKNESS_UM = 800
    SCALE = 0.001  # 800 µm -> 0.8 geometric units

    def __init__(self, root):
        self.root = root
        self.root.title("Separated Segments Viewer with Hover Magnifier")

        self.thickness_um = {
            1: 80, 2: 80, 3: 80, 4: 80,
            5: 80, 6: 80, 7: 80, 8: 80,
            9: 60, 10: 60, 11: 60, 12: 60
        }

        # magnifier settings
        self.zoom_factor = 3.0
        self.zoom_window_size = 2.0  # half-size of inspected square in data units

        self.build_ui()
        self.draw_shape()
        self.update_zoom_panel(None, None)

    # -----------------------------
    # Geometry helpers
    # -----------------------------
    @staticmethod
    def clamp_um(value):
        return max(0.0, min(float(value), ShapeApp.MAX_THICKNESS_UM))

    def um_to_geo(self, value_um):
        return self.clamp_um(value_um) * self.SCALE

    @staticmethod
    def horizontal_segment_polygon(x1, x2, y_outer, thickness_geo):
        return np.array([
            [x1, y_outer],
            [x2, y_outer],
            [x2, y_outer - thickness_geo],
            [x1, y_outer - thickness_geo]
        ], dtype=float)

    @staticmethod
    def vertical_segment_polygon(x_outer, y1, y2, thickness_geo):
        return np.array([
            [x_outer, y1],
            [x_outer, y2],
            [x_outer - thickness_geo, y2],
            [x_outer - thickness_geo, y1]
        ], dtype=float)

    @staticmethod
    def ring_sector_polygon(cx, cy, r_outer, thickness_geo, start_deg, end_deg, n=160):
        r_inner = r_outer - thickness_geo
        if r_inner <= 0:
            r_inner = 0.05

        theta = np.linspace(np.deg2rad(start_deg), np.deg2rad(end_deg), n)

        x_outer = cx + r_outer * np.cos(theta)
        y_outer = cy + r_outer * np.sin(theta)

        x_inner = cx + r_inner * np.cos(theta[::-1])
        y_inner = cy + r_inner * np.sin(theta[::-1])

        x = np.concatenate([x_outer, x_inner])
        y = np.concatenate([y_outer, y_inner])

        return np.column_stack([x, y])

    # -----------------------------
    # Drawing helpers
    # -----------------------------
    def add_patch(self, ax, points, color="navy"):
        ax.add_patch(Polygon(points, closed=True, facecolor=color, edgecolor=color))

    def add_label(self, ax, x, y, text, fontsize=14):
        ax.text(x, y, str(text), fontsize=fontsize, color="navy", ha="center", va="center")

    def draw_segments_on_axis(self, ax):
        """
        Draw the full shape on the given axis using current thicknesses.
        """
        tg = {k: self.um_to_geo(v) for k, v in self.thickness_um.items()}
        label_size = 14

        # --- Segment 1
        seg1 = self.ring_sector_polygon(
            cx=0.6, cy=8.0,
            r_outer=1.4,
            thickness_geo=tg[1],
            start_deg=230, end_deg=360
        )
        self.add_patch(ax, seg1)
        self.add_label(ax, -0.8, 6.9, 1, fontsize=label_size)

        # --- Segment 2
        seg2 = self.horizontal_segment_polygon(
            x1=3.0, x2=6.2, y_outer=8.0,
            thickness_geo=tg[2]
        )
        self.add_patch(ax, seg2)
        self.add_label(ax, 4.6, 8.7, 2, fontsize=label_size)

        # --- Segment 3
        seg3 = self.ring_sector_polygon(
            cx=9.0, cy=8.0,
            r_outer=2.0,
            thickness_geo=tg[3],
            start_deg=180, end_deg=360
        )
        self.add_patch(ax, seg3)
        self.add_label(ax, 9.0, 7.1, 3, fontsize=label_size)

        # --- Segment 4
        seg4 = self.horizontal_segment_polygon(
            x1=11.8, x2=15.2, y_outer=8.0,
            thickness_geo=tg[4]
        )
        self.add_patch(ax, seg4)
        self.add_label(ax, 13.5, 8.7, 4, fontsize=label_size)

        # --- Segment 5
        seg5 = self.ring_sector_polygon(
            cx=17.9, cy=6.9,
            r_outer=1.4,
            thickness_geo=tg[5],
            start_deg=150, end_deg=330
        )
        self.add_patch(ax, seg5)
        self.add_label(ax, 19.8, 6.9, 5, fontsize=label_size)

        # --- Segment 6
        seg6 = self.vertical_segment_polygon(
            x_outer=20.0, y1=6.0, y2=0.4,
            thickness_geo=tg[6]
        )
        self.add_patch(ax, seg6)
        self.add_label(ax, 21.1, 3.2, 6, fontsize=label_size)

        # --- Segment 7
        seg7 = self.ring_sector_polygon(
            cx=19.2, cy=-1.4,
            r_outer=1.7,
            thickness_geo=tg[7],
            start_deg=90, end_deg=270
        )
        self.add_patch(ax, seg7)
        self.add_label(ax, 21.2, -1.4, 7, fontsize=label_size)

        # --- Segment 8
        seg8 = self.vertical_segment_polygon(
            x_outer=20.0, y1=-3.4, y2=-8.8,
            thickness_geo=tg[8]
        )
        self.add_patch(ax, seg8)
        self.add_label(ax, 21.1, -6.2, 8, fontsize=label_size)

        # --- Inner circle
        inner_cx = 9.0
        inner_cy = -1.8
        inner_r_outer = 1.8

        seg9 = self.ring_sector_polygon(
            cx=inner_cx, cy=inner_cy,
            r_outer=inner_r_outer,
            thickness_geo=tg[9],
            start_deg=50, end_deg=130
        )
        self.add_patch(ax, seg9)
        self.add_label(ax, inner_cx, inner_cy + 2.4, 9, fontsize=label_size)

        seg10 = self.ring_sector_polygon(
            cx=inner_cx, cy=inner_cy,
            r_outer=inner_r_outer,
            thickness_geo=tg[10],
            start_deg=-40, end_deg=40
        )
        self.add_patch(ax, seg10)
        self.add_label(ax, inner_cx + 2.1, inner_cy, 10, fontsize=label_size)

        seg11 = self.ring_sector_polygon(
            cx=inner_cx, cy=inner_cy,
            r_outer=inner_r_outer,
            thickness_geo=tg[11],
            start_deg=230, end_deg=310
        )
        self.add_patch(ax, seg11)
        self.add_label(ax, inner_cx, inner_cy - 2.4, 11, fontsize=label_size)

        seg12 = self.ring_sector_polygon(
            cx=inner_cx, cy=inner_cy,
            r_outer=inner_r_outer,
            thickness_geo=tg[12],
            start_deg=140, end_deg=220
        )
        self.add_patch(ax, seg12)
        self.add_label(ax, inner_cx - 2.1, inner_cy, 12, fontsize=label_size)

    def setup_main_axis(self):
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-2.5, 23.0)
        self.ax.set_ylim(-10.5, 10.0)
        self.ax.set_title("Main View")
        self.ax.grid(True, alpha=0.35)

    def setup_zoom_axis(self, center_x, center_y):
        half_w = self.zoom_window_size
        half_h = self.zoom_window_size

        # zoom panel shows a smaller data window, so it looks enlarged
        view_half_w = half_w / self.zoom_factor
        view_half_h = half_h / self.zoom_factor

        self.ax_zoom.set_aspect("equal")
        self.ax_zoom.set_xlim(center_x - view_half_w, center_x + view_half_w)
        self.ax_zoom.set_ylim(center_y - view_half_h, center_y + view_half_h)
        self.ax_zoom.set_title(f"Zoom x{self.zoom_factor:g}")
        self.ax_zoom.grid(True, alpha=0.35)

    # -----------------------------
    # Main rendering
    # -----------------------------
    def draw_shape(self):
        self.ax.clear()
        self.ax_zoom.clear()

        self.draw_segments_on_axis(self.ax)
        self.setup_main_axis()

        self.canvas.draw_idle()

    def update_zoom_panel(self, x, y):
        self.ax_zoom.clear()
        self.draw_segments_on_axis(self.ax_zoom)

        if x is None or y is None:
            # default zoom center
            x, y = 9.0, 8.0

        self.setup_zoom_axis(x, y)
        self.canvas.draw_idle()

    # -----------------------------
    # Mouse hover magnifier
    # -----------------------------
    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = event.xdata
        y = event.ydata

        # redraw main axis with highlight rectangle
        self.ax.clear()
        self.draw_segments_on_axis(self.ax)
        self.setup_main_axis()

        inspect_size = 2 * self.zoom_window_size
        rect = Rectangle(
            (x - self.zoom_window_size, y - self.zoom_window_size),
            inspect_size,
            inspect_size,
            fill=False,
            edgecolor="crimson",
            linewidth=1.5
        )
        self.ax.add_patch(rect)

        # update zoom panel
        self.update_zoom_panel(x, y)

    # -----------------------------
    # UI
    # -----------------------------
    def build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(main, padding=(12, 0, 0, 0))
        right.pack(side="right", fill="y")

        # two panels: main + zoom
        self.fig, (self.ax, self.ax_zoom) = plt.subplots(1, 2, figsize=(13, 6))
        self.fig.tight_layout(pad=2.0)

        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        ttk.Label(right, text="Segment Thickness (µm)", font=("Arial", 13, "bold")).pack(pady=(0, 10))

        self.entries = {}

        for seg_id in range(1, 13):
            row = ttk.Frame(right)
            row.pack(fill="x", pady=3)

            ttk.Label(row, text=f"Segment {seg_id}", width=12).pack(side="left")

            entry = ttk.Entry(row, width=10)
            entry.insert(0, str(self.thickness_um[seg_id]))
            entry.pack(side="left")
            self.entries[seg_id] = entry

        ttk.Label(
            right,
            text=f"Allowed range: 0 to {self.MAX_THICKNESS_UM} µm",
            foreground="darkgreen"
        ).pack(pady=(10, 6))

        # magnifier controls
        zoom_factor_frame = ttk.Frame(right)
        zoom_factor_frame.pack(fill="x", pady=(8, 4))

        ttk.Label(zoom_factor_frame, text="Zoom factor", width=12).pack(side="left")
        self.zoom_factor_entry = ttk.Entry(zoom_factor_frame, width=10)
        self.zoom_factor_entry.insert(0, str(self.zoom_factor))
        self.zoom_factor_entry.pack(side="left")

        zoom_window_frame = ttk.Frame(right)
        zoom_window_frame.pack(fill="x", pady=(4, 8))

        ttk.Label(zoom_window_frame, text="Window size", width=12).pack(side="left")
        self.zoom_window_entry = ttk.Entry(zoom_window_frame, width=10)
        self.zoom_window_entry.insert(0, str(self.zoom_window_size))
        self.zoom_window_entry.pack(side="left")

        ttk.Button(right, text="Update Shape", command=self.update_values).pack(fill="x", pady=4)
        ttk.Button(right, text="Set Magnifier", command=self.update_magnifier_settings).pack(fill="x", pady=4)
        ttk.Button(right, text="Reset", command=self.reset_values).pack(fill="x", pady=4)

        info = (
            "Hover magnifier:\n"
            "- move mouse over Main View\n"
            "- right panel shows enlarged area\n"
            "- red box shows inspected region"
        )
        ttk.Label(
            right,
            text=info,
            foreground="gray",
            wraplength=230,
            justify="left"
        ).pack(pady=(10, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(
            right,
            textvariable=self.status_var,
            foreground="darkred",
            wraplength=230,
            justify="left"
        ).pack(pady=(10, 0))

    # -----------------------------
    # Actions
    # -----------------------------
    def update_magnifier_settings(self):
        try:
            zf = float(self.zoom_factor_entry.get().strip())
            ws = float(self.zoom_window_entry.get().strip())

            if zf <= 0:
                raise ValueError("Zoom factor must be greater than zero.")
            if ws <= 0:
                raise ValueError("Window size must be greater than zero.")

            self.zoom_factor = zf
            self.zoom_window_size = ws

            self.draw_shape()
            self.update_zoom_panel(None, None)

            self.status_var.set(
                f"Magnifier updated: zoom x{self.zoom_factor:g}, window {self.zoom_window_size:g}"
            )
        except ValueError as e:
            messagebox.showerror("Invalid Magnifier Settings", str(e))
            self.status_var.set("Invalid magnifier settings.")

    def update_values(self):
        try:
            changed = []
            for seg_id in range(1, 13):
                raw = self.entries[seg_id].get().strip()
                if raw == "":
                    raise ValueError(f"Segment {seg_id}: empty value")

                requested = float(raw)
                applied = self.clamp_um(requested)
                self.thickness_um[seg_id] = applied

                self.entries[seg_id].delete(0, tk.END)
                self.entries[seg_id].insert(0, str(int(applied) if applied.is_integer() else applied))

                if requested != applied:
                    changed.append(f"S{seg_id}: {requested} → {applied}")

            self.draw_shape()
            self.update_zoom_panel(None, None)

            if changed:
                self.status_var.set(
                    "Some values were clamped to 0..800 µm:\n" +
                    ", ".join(changed[:5]) +
                    (" ..." if len(changed) > 5 else "")
                )
            else:
                self.status_var.set("Shape updated successfully.")
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            self.status_var.set("Invalid input.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.status_var.set(f"Unexpected error: {e}")

    def reset_values(self):
        defaults = {
            1: 80, 2: 80, 3: 80, 4: 80,
            5: 80, 6: 80, 7: 80, 8: 80,
            9: 60, 10: 60, 11: 60, 12: 60
        }
        self.thickness_um = defaults.copy()
        self.zoom_factor = 3.0
        self.zoom_window_size = 2.0

        for seg_id in range(1, 13):
            self.entries[seg_id].delete(0, tk.END)
            self.entries[seg_id].insert(0, str(defaults[seg_id]))

        self.zoom_factor_entry.delete(0, tk.END)
        self.zoom_factor_entry.insert(0, str(self.zoom_factor))

        self.zoom_window_entry.delete(0, tk.END)
        self.zoom_window_entry.insert(0, str(self.zoom_window_size))

        self.draw_shape()
        self.update_zoom_panel(None, None)
        self.status_var.set("Reset to default values.")


if __name__ == "__main__":
    root = tk.Tk()
    app = ShapeApp(root)
    root.mainloop()