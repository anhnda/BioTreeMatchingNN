import numpy as np
from sklearn.linear_model import LinearRegression

def get_projection_onto_lines(w, c, points):
    y0 = points[:, 1]
    x0 = points[:, 0]
    t = (w * (y0 - c) + x0) / (1 + w ** 2)
    y = w * t + c
    projected_points = np.hstack([t.reshape(-1,1), y.reshape(-1,1)])
    return projected_points

def get_projection_lr(points):
    model = LinearRegression().fit(points[:,0].reshape(-1,1), points[:,1])
    w = model.coef_
    c = model.intercept_
    return get_projection_onto_lines(w,c,points), w, c

def get_online_position(points):
    projected_positions, w, c = get_projection_lr(points)
    xmin = -100
    ymin = xmin * w[0] + c

    pmin = [xmin, ymin]
    print(pmin)
    pmin = np.asarray(pmin)
    offset_vecs = projected_positions - pmin
    distances = np.sqrt(np.sum(offset_vecs * offset_vecs, axis=1))
    sorted_indices = np.argsort(distances)
    assert distances[sorted_indices[0]] <= distances[sorted_indices[-1]]
    min_distance = distances[sorted_indices[0]]
    distances -= min_distance
    sorted_distances = distances[sorted_indices]
    online_positions = np.zeros(len(sorted_distances))
    for i, v in enumerate(sorted_indices):
        online_positions[v] = sorted_distances[i]


    return online_positions
if __name__ == "__main__":
    w = 1
    c = 0
    points = np.hstack([np.zeros(10).reshape(-1,1), np.arange(0, 10).reshape(-1,1)])
    print(points.shape)
    print(get_projection_onto_lines(w, c, points))
