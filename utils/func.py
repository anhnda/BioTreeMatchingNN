import colorsys
import matplotlib


def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s=s)


def get_scale_light_color_list(color="orangered"):
    color = matplotlib.colors.ColorConverter.to_rgb(color)
    rgbs = [scale_lightness(color, scale) for scale in [ 0.75, 1, 1.5, 1.95]]
    return rgbs


def get_insert_dict_index(d, k, offset=0):
    try:
        idx = d[k]
    except:
        idx = len(d) + offset
        d[k] = idx

    return idx


def get_index_dict(d, k, v=-1):
    try:
        v = d[k]
    except:
        pass
    return v
