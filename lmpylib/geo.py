import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from landez import ImageExporter
from landez.filters import GrayScale
from landez.sources import MBTilesReader
import itertools
import math


def trajectory(lat, lng, trip_id=None, max_sample=100, line_style="solid", line_width=2, color="gray", marker="o", marker_size=5, mark_od=True, show_ticks=False, show=False):
    if len(lng) != len(lat):
        raise Exception("Length of lng array and lat array must be same.")
    if trip_id is not None:
        if len(lng) == len(trip_id):
            unique_trips = np.unique(trip_id)
            n_colors = min(len(unique_trips), 20)
            for t, trip in enumerate(unique_trips):
                hue = (t % n_colors) / n_colors
                bright = int(t / 20) % 3 * 0.2 + 0.6
                # bright = (t % 3) * 0.2 + 0.6
                # hue = (int(t / 3) % n_colors) / n_colors
                rgb = np.floor(colors.hsv_to_rgb([hue, 0.9, bright])*255).astype(int)
                crl = "#" + "".join(['{:02x}'.format(num) for num in rgb])
                trajectory(lat[trip_id == trip], lng[trip_id == trip], trip_id=None, max_sample=max_sample,
                           line_style=line_style, line_width=line_width, color=crl, marker=marker,
                           marker_size=marker_size, mark_od=mark_od, show_ticks=show_ticks, show=False)
            return
        else:
            raise Exception("Length of trip_id array and points array must be same.")

    points = np.array([lng, lat]).transpose()
    n_points = len(points)

    if (max_sample is not None) and (max_sample > 0) and (n_points > max_sample):
        indices = np.random.choice(list(range(n_points)), max_sample, replace=False)
        indices.sort()
        points = points[indices, :]
        n_points = max_sample

    if n_points == 1:
        plt.scatter(points[0][0], points[0][1], marker=marker, s=marker_size ** 2, color=color)
    elif n_points > 1:
        section_lng, section_lat = [0, 0], [0, 0]
        for i, point in enumerate(points):
            section_lng[0] = section_lng[1]
            section_lat[0] = section_lat[1]
            section_lng[1] = point[0]
            section_lat[1] = point[1]
            if i > 0:
                if mark_od and ((i == 1) or (i == n_points - 1)):
                    plt.plot(section_lng, section_lat, linewidth=line_width, linestyle=line_style, color=color, zorder=1)
                else:
                    plt.plot(section_lng, section_lat, linewidth=line_width, linestyle=line_style, color=color,
                         marker=marker, markersize=marker_size, zorder=1)
        if mark_od:
            plt.scatter(points[0][0], points[0][1], marker=">", s=marker_size**2*4, color="green", edgecolors="#225316", zorder=2)
            plt.scatter(points[n_points-1][0], points[n_points-1][1], marker="s", s=marker_size**2*3, c="red", edgecolors="#a72e1a", zorder=2)

    if not show_ticks:
        plt.xticks([])
        plt.yticks([])

    if show:
        plt.show()


trn = np.array([[1.340116, 103.694897],
[1.338571, 103.695541],
[1.334882, 103.696442],
[1.334839, 103.698974],
[1.337670, 103.698760],
[1.341618, 103.696185],
[1.340674, 103.694253],
[1.345436, 103.690777]])

trn = np.insert(trn, 1, np.array([np.random.normal(trn[0, 0], 0.0001, 3), np.random.normal(trn[0, 1], 0.0001, 3)]).transpose(), axis=0)
print(len(trn))
trn = np.insert(trn, 8, np.array([np.random.normal(trn[7, 0], 0.0002, 2), np.random.normal(trn[7, 1], 0.0002, 2)]).transpose(), axis=0)
print(len(trn))
trn = np.insert(trn, len(trn)-1, np.array([np.random.normal(trn[len(trn)-1, 0], 0.0001, 100), np.random.normal(trn[len(trn)-1, 1], 0.0001, 100)]).transpose(), axis=0)
print(len(trn))

# trn = np.array([[1.340116, 103.694897], [1.340674, 103.694253]])
trip_id = [1]*7 + [2]*106
trajectory(trn[:, 0], trn[:, 1], trip_id, mark_od=True, max_sample=0)

gmap3 = gmplot.GoogleMapPlotter(1.34, 103.69, 13)

ie = ImageExporter(mbtiles_file="/Users/liming/Downloads/2017-07-03_asia_malaysia-singapore-brunei.mbtiles")
ie.add_filter(GrayScale())
ie.export_image(bbox=(103.59, 1.201, 104.05, 1.49), zoomlevel=5, imagepath="map0.png")

from io import BytesIO, StringIO
from PIL.Image import ID, OPEN
import logging
logging.basicConfig(level=logging.DEBUG)
mbreader = MBTilesReader("/Users/liming/Downloads/2017-07-03_asia_malaysia-singapore-brunei.mbtiles")
grid = ie.grid_tiles((103.59, 1.201, 104.05, 1.49), 3)
data = mbreader.tile(3, 6, 3)

with open('tile.txt', 'wb') as out:
    out.write(data)

fp = BytesIO(data)
prefix = fp.read(16)
factory, accept = OPEN["JPEG"]
accept(prefix)
ID
OPEN
Image.open(fp)
data = ie._tile_image(ie.tile((3, 6, 3)))