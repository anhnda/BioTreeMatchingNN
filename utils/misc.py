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
    return get_projection_onto_lines(w,c,points)

if __name__ == "__main__":
    w = 1
    c = 0
    points = np.hstack([np.zeros(10).reshape(-1,1), np.arange(0, 10).reshape(-1,1)])
    print(points.shape)
    print(get_projection_onto_lines(w, c, points))
