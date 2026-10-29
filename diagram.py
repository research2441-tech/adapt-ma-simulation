"""Small grayscale block-diagram engine for IEEE figures.

Boxes are placed explicitly in data units (inches). Connectors attach to box edges, and the
engine refuses to save a figure in which boxes overlap, labels overflow their boxes, or a
connector label collides with a box.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.transforms import Bbox

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'Times New Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7.5})


class Diagram:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w); self.ax.set_ylim(0, h); self.ax.axis('off')
        self.w, self.h = w, h
        self.boxes, self.texts, self.labels = {}, [], []

    def box(self, name, x, y, w, h, text, fill='white', lw=0.8, ls='-', bold=False, size=7.0, round_=0.04):
        p = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=f'round,pad=0,rounding_size={round_}',
                           fc=fill, ec='black', lw=lw, ls=ls, zorder=2)
        self.ax.add_patch(p)
        t = self.ax.text(x, y, text, ha='center', va='center', fontsize=size, zorder=3,
                         fontweight='bold' if bold else 'normal', linespacing=1.15)
        self.boxes[name] = (x, y, w, h)
        self.texts.append((name, t))
        return name

    def group(self, x, y, w, h, title, size=7.0):
        """Dashed grouping frame with a title tab; does not take part in the overlap test."""
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.06', fc='none', ec='0.35',
                           lw=0.7, ls=(0, (4, 2)), zorder=1)
        self.ax.add_patch(p)
        t = self.ax.text(x + 0.08, y + h - 0.06, title, ha='left', va='top', fontsize=size, style='italic',
                         color='0.15', zorder=3, bbox=dict(fc='white', ec='none', pad=0.6))
        self.labels.append(t)

    def anchor(self, name, side, frac=0.5):
        x, y, w, h = self.boxes[name]
        if side == 'l':
            return (x - w / 2, y - h / 2 + frac * h)
        if side == 'r':
            return (x + w / 2, y - h / 2 + frac * h)
        if side == 't':
            return (x - w / 2 + frac * w, y + h / 2)
        if side == 'b':
            return (x - w / 2 + frac * w, y - h / 2)
        raise ValueError(side)

    def arrow(self, a, b, dashed=False, lw=0.8, label=None, lpos=0.5, loff=(0, 0.07), conn='arc3,rad=0', size=6.5,
              head='-|>'):
        p = FancyArrowPatch(a, b, arrowstyle=head, mutation_scale=7, lw=lw, color='black',
                            ls=(0, (3, 2)) if dashed else '-', connectionstyle=conn, zorder=4,
                            shrinkA=0, shrinkB=0)
        self.ax.add_patch(p)
        if label:
            lx = a[0] + (b[0] - a[0]) * lpos + loff[0]; ly = a[1] + (b[1] - a[1]) * lpos + loff[1]
            t = self.ax.text(lx, ly, label, ha='center', va='center', fontsize=size, zorder=5,
                             bbox=dict(fc='white', ec='none', pad=0.5), style='italic')
            self.labels.append(t)

    def poly(self, pts, dashed=False, lw=0.8, label=None, lseg=0, loff=(0, 0.07), size=6.5):
        """Orthogonal multi-segment connector ending in an arrow head."""
        for i in range(len(pts) - 2):
            self.ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]], color='black', lw=lw,
                         ls=(0, (3, 2)) if dashed else '-', zorder=4, solid_capstyle='butt')
        self.arrow(pts[-2], pts[-1], dashed=dashed, lw=lw)
        if label:
            a, b = pts[lseg], pts[lseg + 1]
            t = self.ax.text((a[0] + b[0]) / 2 + loff[0], (a[1] + b[1]) / 2 + loff[1], label, ha='center',
                             va='center', fontsize=size, zorder=5, bbox=dict(fc='white', ec='none', pad=0.5),
                             style='italic')
            self.labels.append(t)

    def note(self, x, y, text, size=6.5, ha='center', style='normal'):
        t = self.ax.text(x, y, text, ha=ha, va='center', fontsize=size, zorder=5, style=style)
        self.labels.append(t)

    # ------------------------------------------------------------------ checks
    def check(self, gap=0.06):
        self.fig.canvas.draw()
        r = self.fig.canvas.get_renderer()
        inv = self.ax.transData.inverted()
        names = list(self.boxes)
        for i in range(len(names)):
            xi, yi, wi, hi = self.boxes[names[i]]
            for j in range(i + 1, len(names)):
                xj, yj, wj, hj = self.boxes[names[j]]
                if abs(xi - xj) < (wi + wj) / 2 + gap and abs(yi - yj) < (hi + hj) / 2 + gap:
                    raise RuntimeError(f'boxes overlap or touch: {names[i]} / {names[j]}')
        for name, t in self.texts:
            bb = Bbox(inv.transform(t.get_window_extent(r)))
            x, y, w, h = self.boxes[name]
            if bb.width > w - 0.06 or bb.height > h - 0.04:
                raise RuntimeError(f'text overflows box {name}: {bb.width:.2f}x{bb.height:.2f} in {w}x{h}')
        for t in self.labels:
            bb = Bbox(inv.transform(t.get_window_extent(r)))
            if bb.x0 < 0 or bb.x1 > self.w or bb.y0 < 0 or bb.y1 > self.h:
                raise RuntimeError(f'label outside canvas: {t.get_text()}')
            for name, (x, y, w, h) in self.boxes.items():
                if bb.x1 > x - w / 2 and bb.x0 < x + w / 2 and bb.y1 > y - h / 2 and bb.y0 < y + h / 2:
                    raise RuntimeError(f'label "{t.get_text()}" collides with box {name}')
        for i in range(len(self.labels)):
            bi = Bbox(inv.transform(self.labels[i].get_window_extent(r)))
            for j in range(i + 1, len(self.labels)):
                bj = Bbox(inv.transform(self.labels[j].get_window_extent(r)))
                if bi.overlaps(bj):
                    raise RuntimeError(f'labels overlap: {self.labels[i].get_text()} / {self.labels[j].get_text()}')

    def save(self, path, dpi=600):
        self.check()
        self.fig.savefig(path, dpi=dpi, facecolor='white')
        plt.close(self.fig)
