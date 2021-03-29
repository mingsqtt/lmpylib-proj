import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib
from matplotlib.patches import Polygon
from shapely import geometry
# from mpl_toolkits.basemap import Basemap
from matplotlib.collections import PatchCollection
# from landez import ImageExporter
# from landez.filters import GrayScale
# from landez.sources import MBTilesReader
import itertools
import math
from .core import shift_down, shift_up
from .core import movoper


def plt_trajectory(lat, lng, trip_id=None, max_sample=100, line_style="solid", line_width=2, color="gray", marker="o", marker_size=5, mark_od=True, show_ticks=False, show=False):
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
                plt_trajectory(lat[trip_id == trip], lng[trip_id == trip], trip_id=None, max_sample=max_sample,
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


# According to the NMEA Standard, Latitude and Longitude are output in the format Degrees, Minutes and
# (Decimal) Fractions of Minutes. To convert to Degrees and Fractions of Degrees, or Degrees, Minutes, Seconds
# and Fractions of seconds, the 'Minutes' and 'Fractional Minutes' parts need to be converted. In other words: If
# the GPS Receiver reports a Latitude of 4717.112671 North and Longitude of 00833.914843 East, this is
#
# Latitude 47 Degrees, 17.112671 Minutes
# Longitude 8 Degrees, 33.914843 Minutes
# or
# Latitude 47 Degrees, 17 Minutes, 6.76026 Seconds
# Longitude 8 Degrees, 33 Minutes, 54.89058 Seconds
# or
# Latitude 47.28521118 Degrees
# Longitude 8.56524738 Degrees
def nmea_to_degree_fraction(nmea_value):
    degree = np.floor(nmea_value / 100)
    min_frac = nmea_value - degree * 100
    return np.round(degree + min_frac / 60, 7)


def effective_displacement(points, uom="m"):
    """
    :param points: (n, prev_lat_lng, current_lat_lng) or (n, prev_lat_lng, current_lat_lng, next_lat_lng) or (n, prev_lat, prev_lng, curr_lat, curr_lng) or (n, prev_lat, prev_lng, curr_lat, curr_lng, next_lat, next_lng)
    :param uom: "m" or "km"
    :return: (distance, orientation_angle)
    """
    assert (((len(points.shape) == 3) or (len(points.shape) == 4)) and (points.shape[1] == 2) and (points.shape[2] == 2)) or (
            (len(points.shape) == 2) and ((points.shape[1] == 4) or (
            points.shape[1] == 6))), "points must have shape of (n, 2, 2) or (n, 2, 2, 2) or (n, 4) or (n, 6)"
    assert (uom == "m") or (uom == "km"), "uom must be km or m"
    if len(points.shape) == 3:
        vec = np.sum(points[:, 1, :] - points[:, 0, :], axis=0)
        y = vec[0]
        x = vec[1]
    else:
        y = np.sum(points[:, 2] - points[:, 0])
        x = np.sum(points[:, 3] - points[:, 1])

    angles = np.arctan2(y, x)
    kms = np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)

    if uom == "km":
        return kms, angles
    elif uom == "m":
        return np.round(kms * 1000, 0), angles
    else:
        return None, None


def displacement(points, uom="km", include_orient_change=False):
    """
    :param points: (n, prev_lat_lng, current_lat_lng) or (n, prev_lat_lng, current_lat_lng, next_lat_lng) or (n, prev_lat, prev_lng, curr_lat, curr_lng) or (n, prev_lat, prev_lng, curr_lat, curr_lng, next_lat, next_lng)
    :param uom: "m" or "km"
    :param include_orient_change: include cosine of the orientation angle change in the output
    :return: (distance, orientation_angle) or (distance, orientation_angle, cosine_angle_change)
    """
    if include_orient_change:
        assert ((len(points.shape) == 4) and (points.shape[1] == 2) and (points.shape[2] == 2) and (points.shape[3] == 2)) or (
                (len(points.shape) == 2) and (points.shape[1] == 6)), "points must have shape of (n, 2, 2, 2) or (n, 6)"
    else:
        assert (((len(points.shape) == 3) or (len(points.shape) == 4)) and (points.shape[1] == 2) and (points.shape[2] == 2)) or (
                       (len(points.shape) == 2) and ((points.shape[1] == 4) or (
                       points.shape[1] == 6))), "points must have shape of (n, 2, 2) or (n, 2, 2, 2) or (n, 4) or (n, 6)"
    assert (uom == "m") or (uom == "km"), "uom must be km or m"
    if len(points.shape) == 2:
        y1 = points[:, 2] - points[:, 0]
        x1 = points[:, 3] - points[:, 1]
    else:
        vec = points[:, 1, :] - points[:, 0, :]
        y1 = vec[:, 0]
        x1 = vec[:, 1]

    angles = np.arctan2(y1, x1)
    kms = np.round((x1 ** 2 + y1 ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)

    if include_orient_change:
        if len(points.shape) == 2:
            y2 = points[:, 4] - points[:, 2]
            x2 = points[:, 5] - points[:, 3]
            next_next_lat = np.array(shift_up(points[:, 4]))
            next_next_lng = np.array(shift_up(points[:, 5]))
            y3 = next_next_lat - points[:, 2]
            x3 = next_next_lng - points[:, 3]
        else:
            vec = points[:, 2, :] - points[:, 1, :]
            y2 = vec[:, 0]
            x2 = vec[:, 1]
            next_next_lat = np.array(shift_up(points[:, 2, 0]))
            next_next_lng = np.array(shift_up(points[:, 2, 1]))
            y3 = next_next_lat - points[:, 1, 0]
            x3 = next_next_lng - points[:, 1, 1]

        next_dist = np.array(shift_up(kms, fill_tail_with=0))

        sim_2_1 = (x1 * x2 + y1 * y2) / ((x1 ** 2 + y1 ** 2) ** 0.5 * (x2 ** 2 + y2 ** 2) ** 0.5)
        sim_2_1[pd.isna(sim_2_1)] = 1

        sim_3_1 = (x1 * x3 + y1 * y3) / ((x1 ** 2 + y1 ** 2) ** 0.5 * (x3 ** 2 + y3 ** 2) ** 0.5)
        sim_3_1[pd.isna(sim_3_1)] = 1

        override_filt = next_dist < 0.03
        sim_2_1[override_filt] = np.minimum(sim_2_1[override_filt], sim_3_1[override_filt])

        if uom == "km":
            return kms, angles, sim_2_1
        elif uom == "m":
            return np.round(kms * 1000, 0), angles, sim_2_1
        return None, None, None
    else:
        if uom == "km":
            return kms, angles
        elif uom == "m":
            return np.round(kms * 1000, 0), angles
        return None, None


def distance(lat_prev, lng_prev, lat, lng, uom="km"):
    assert (uom == "m") or (uom == "km"), "uom must be km or m"
    if type(lat_prev) == np.ndarray:
        assert (len(lat_prev) == len(lng_prev)) and (len(lat_prev) == len(lat)) and (len(lat_prev) == len(lng)), "array sizes are not the same"
    else:
        if (lat_prev == 0) or (lng_prev == 0) or (lat == 0) or (lng == 0):
            return 0
    y = lat - lat_prev
    x = lng - lng_prev
    if uom == "km":
        return np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
    elif uom == "m":
        return np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 * 1000 / 2 / 90, 0)
    else:
        return None


def distance_matrix(lat_a, lng_a, lat_b, lng_b, uom="km"):
    if type(lat_a) == list:
        lat_a = np.array(lat_a)
    if type(lng_a) == list:
        lng_a = np.array(lng_a)
    if type(lat_b) == list:
        lat_b = np.array(lat_b)
    if type(lng_b) == list:
        lng_b = np.array(lng_b)
    assert (type(lat_a) == np.ndarray) and (type(lng_a) == np.ndarray) and (type(lat_b) == np.ndarray) and (
                type(lng_b) == np.ndarray), "inputs must be 1-d ndarray"
    assert (len(lat_a) == len(lng_a)) and (len(lat_b) == len(lng_b)), "array sizes are not the same"
    assert (uom == "m") or (uom == "km"), "uom must be km or m"
    mtx_a2b_lat = np.repeat(lat_a, len(lat_b)).reshape((-1, len(lat_b)))
    mtx_a2b_lng = np.repeat(lng_a, len(lng_b)).reshape((-1, len(lng_b)))
    mtx_b2a_lat = np.repeat(lat_b.reshape((-1, 1)), len(lat_a), axis=1).transpose()
    mtx_b2a_lng = np.repeat(lng_b.reshape((-1, 1)), len(lng_a), axis=1).transpose()
    y = mtx_a2b_lat - mtx_b2a_lat
    x = mtx_a2b_lng - mtx_b2a_lng
    if uom == "km":
        return np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 / 2 / 90, 3)
    elif uom == "m":
        return np.round((x ** 2 + y ** 2) ** 0.5 * np.pi * 6371 * 1000 / 2 / 90, 0)
    else:
        return None


def orientation(lat_prev, lng_prev, lat, lng):
    if type(lat_prev) == np.ndarray:
        assert (len(lat_prev) == len(lng_prev)) and (len(lat_prev) == len(lat)) and (len(lat_prev) == len(lng)), "array sizes are not the same"
    else:
        if (lat_prev == 0) or (lng_prev == 0) or (lat == 0) or (lng == 0):
            return 0
    y = lat - lat_prev
    x = lng - lng_prev
    return np.arctan2(y, x)


def _get_km_dist(points):
    return distance(points[0], points[1], points[2], points[3])


def _create_speed_features(ordered_time, lat, lng, include_slope, include_prev, include_next):
    n_pos = len(ordered_time)

    time_delta = np.round((ordered_time.values - pd.Series(shift_down(ordered_time)).values) / np.timedelta64(1, 's'), 1)

    # crossed_pos = np.array(range(n_pos*2), dtype=float)
    # odd = crossed_pos.astype(int) % 2 == 1
    # even = crossed_pos.astype(int) % 2 == 0
    # crossed_pos[even] = lat
    # crossed_pos[odd] = lng
    # km_dist = np.concatenate([[0], np.abs(movoper(crossed_pos, _get_km_dist, 4, "s", "valid", stripe=2, return_valid_only=True))])

    prev_lat = shift_down(lat)
    prev_lng = shift_down(lng)
    km_dist, orient = displacement(np.stack([prev_lat, prev_lng, lat, lng]).transpose())

    if len(time_delta) > 1:
        time_delta[0] = time_delta[1]
        km_dist[0] = km_dist[1]
        orient[0] = orient[1]

    orient_prev = np.array(shift_down(orient, fill_head_with="same"))
    km_dist_prev = np.array(shift_down(km_dist, fill_head_with="same"))
    km_dist_prev2 = np.array(shift_down(km_dist_prev, fill_head_with="same"))
    km_dist_slope = np.round(km_dist / (km_dist_prev + 1e-3), 4)
    kmh = km_dist / (np.maximum(time_delta, 1)/3600)
    kmh_prev = np.array(shift_down(kmh, fill_head_with="same"))
    kmh_prev2 = np.array(shift_down(kmh_prev, fill_head_with="same"))
    kmh_slope = np.round(kmh / (kmh_prev + 1e-1), 4)
    kmh_slope2 = np.round(kmh / (kmh_prev2 + 1e-1), 4)
    kmh = np.round(kmh, 1)
    kmh_prev = np.round(kmh_prev, 1)
    kmh_prev2 = np.round(kmh_prev2, 1)

    output = pd.DataFrame({
        "time_del": time_delta,
        "dist": km_dist,
        "orient": np.round(orient, 3),
        "speed": kmh
    })

    if include_prev:
        output["dist_prev"] = km_dist_prev
        output["dist_prev2"] = km_dist_prev2
        output["speed_prev"] = kmh_prev
        output["speed_prev2"] = kmh_prev2
        output["orient_prev"] = orient_prev

    if include_slope:
        output["dist_slope"] = km_dist_slope
        output["speed_slope"] = kmh_slope
        output["speed_slope2"] = kmh_slope2

    if include_next:
        time_to_next_lag = np.round((pd.Series(shift_up(ordered_time, fill_tail_with="same")).values - ordered_time.values) / np.timedelta64(1, 's'), 1)
        if len(time_to_next_lag) > 1:
            try:
                time_to_next_lag[-1] = time_to_next_lag[-2]
            except:
                print(time_to_next_lag)
        output["time_to_next_lag"] = time_to_next_lag
        output["speed_next"] = np.array(shift_up(kmh, fill_tail_with="same"))
        output["dist_next"] = np.array(shift_up(km_dist, fill_tail_with="same"))

    return output


def create_speed_features(ordered_df_copy, ordered_time, lat, lng, ordered_vehicle=None, timezone="Asia/Singapore", slope=True, previous=False, next=False):
    if (type(ordered_time) == str) and (ordered_df_copy is not None):
        ordered_time = ordered_df_copy[ordered_time]
    if type(ordered_time) != pd.core.series.Series:
        ordered_time = pd.Series(ordered_time)
    if ordered_time.dt is None:
        ordered_time = pd.to_datetime(ordered_time)
    if ordered_time.dt.tz is None:
        ordered_time = ordered_time.dt.tz_localize(timezone)

    if (type(lat) == str) and (ordered_df_copy is not None):
        lat = ordered_df_copy[lat]

    if (type(lng) == str) and (ordered_df_copy is not None):
        lng = ordered_df_copy[lng]

    if (ordered_vehicle is not None) and (type(ordered_vehicle) == str) and (ordered_df_copy is not None):
        ordered_vehicle = ordered_df_copy[ordered_vehicle]

    n_pos = len(ordered_time)

    if n_pos == 0:
        return ordered_df_copy
    assert n_pos == len(lat), "ordered_time and lat must have the same size"
    assert n_pos == len(lng), "ordered_time and lng must have the same size"

    if ordered_vehicle is None:
        df_feat = _create_speed_features(ordered_time, lat, lng, slope, previous, next)
    else:
        vehicles = list()
        last_veh = None
        for veh in ordered_vehicle:
            if (last_veh is None) or (veh != last_veh):
                vehicles.append(veh)
                last_veh = veh

        df_feat = None
        for veh in vehicles:
            veh_filt = ordered_vehicle == veh
            df_feat_veh = _create_speed_features(ordered_time[veh_filt], lat[veh_filt], lng[veh_filt], slope, previous, next)
            if df_feat is None:
                df_feat = df_feat_veh
            else:
                df_feat = pd.concat([df_feat, df_feat_veh], ignore_index=True)

    if ordered_df_copy is None:
        return df_feat
    else:
        df_feat.index = ordered_df_copy.index
        return pd.concat([ordered_df_copy, df_feat], axis=1)


def trajectory_steering_distribution(movement_orientations, bins=6, return_prob=True, add_one_smooth=False):
    movement_orientations = np.asarray(movement_orientations)
    assert len(movement_orientations) >= 2, "movement_orientations length must be at least 2"
    prev_orients = np.array(shift_down(movement_orientations))
    angle_delta = np.abs(movement_orientations - prev_orients)
    diff_signs = np.sign(movement_orientations) != np.sign(prev_orients)
    lt_pi = diff_signs & (np.abs(movement_orientations) + np.abs(prev_orients) < np.pi)
    gt_pi = diff_signs & (np.abs(movement_orientations) + np.abs(prev_orients) >= np.pi)
    angle_delta[lt_pi] = np.abs(movement_orientations[lt_pi]) + np.abs(prev_orients[lt_pi])
    angle_delta[gt_pi] = 2*np.pi - (np.abs(movement_orientations[gt_pi]) + np.abs(prev_orients[gt_pi]))
    angle_delta = angle_delta[1:]
    bin_cut = np.pi / bins
    counts = np.zeros(bins)
    for b in range(bins):
        if b < bins - 1:
            counts[b] = np.sum((angle_delta >= bin_cut * b) & (angle_delta < bin_cut * (b + 1)))
        else:
            counts[b] = np.sum(angle_delta >= bin_cut * b)
        if add_one_smooth and return_prob:
            counts[b] = counts[b] + 1
    if return_prob:
        return counts / np.sum(counts)
    else:
        return counts


def get_polygon(coord):
    if (type(coord) == list) or (type(coord) == np.ndarray):
        return geometry.Polygon(list(coord))
    else:
        raise Exception("coord has to be a list of tuple (x, y)")


def point_within(x, y, polygon):
    point = geometry.Point(x, y)
    if (type(polygon) == list) or (type(polygon) == np.ndarray):
        poly = get_polygon(polygon)
        return point.within(poly)
    elif type(polygon) == geometry.Polygon:
        return point.within(polygon)
    else:
        raise Exception("polygon_coord has to be a list of tuple (x, y)")




# basemap = Basemap(llcrnrlon= 75,llcrnrlat=10,urcrnrlon=150,urcrnrlat=55,projection='poly',lon_0 = 116.65,lat_0 = 40.02,ax = ax)

#
# trn = np.array([[1.340116, 103.694897],
# [1.338571, 103.695541],
# [1.334882, 103.696442],
# [1.334839, 103.698974],
# [1.337670, 103.698760],
# [1.341618, 103.696185],
# [1.340674, 103.694253],
# [1.345436, 103.690777]])
#
# trn = np.insert(trn, 1, np.array([np.random.normal(trn[0, 0], 0.0001, 3), np.random.normal(trn[0, 1], 0.0001, 3)]).transpose(), axis=0)
# print(len(trn))
# trn = np.insert(trn, 8, np.array([np.random.normal(trn[7, 0], 0.0002, 2), np.random.normal(trn[7, 1], 0.0002, 2)]).transpose(), axis=0)
# print(len(trn))
# trn = np.insert(trn, len(trn)-1, np.array([np.random.normal(trn[len(trn)-1, 0], 0.0001, 100), np.random.normal(trn[len(trn)-1, 1], 0.0001, 100)]).transpose(), axis=0)
# print(len(trn))
#
# # trn = np.array([[1.340116, 103.694897], [1.340674, 103.694253]])
# trip_id = [1]*7 + [2]*106
# trajectory(trn[:, 0], trn[:, 1], trip_id, mark_od=True, max_sample=0)
#
# gmap3 = gmplot.GoogleMapPlotter(1.34, 103.69, 13)
#
# ie = ImageExporter(mbtiles_file="/Users/liming/Downloads/2017-07-03_asia_malaysia-singapore-brunei.mbtiles")
# ie.add_filter(GrayScale())
# ie.export_image(bbox=(103.59, 1.201, 104.05, 1.49), zoomlevel=5, imagepath="map0.png")
#
# from io import BytesIO, StringIO
# from PIL.Image import ID, OPEN
# import logging
# logging.basicConfig(level=logging.DEBUG)
# mbreader = MBTilesReader("/Users/liming/Downloads/2017-07-03_asia_malaysia-singapore-brunei.mbtiles")
# grid = ie.grid_tiles((103.59, 1.201, 104.05, 1.49), 3)
# data = mbreader.tile(3, 6, 3)
#
# with open('tile.txt', 'wb') as out:
#     out.write(data)
#
# fp = BytesIO(data)
# prefix = fp.read(16)
# factory, accept = OPEN["JPEG"]
# accept(prefix)
# ID
# OPEN
# Image.open(fp)
# data = ie._tile_image(ie.tile((3, 6, 3)))



