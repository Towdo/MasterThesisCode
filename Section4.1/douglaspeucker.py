import numpy as np

def pDistance(x, y, x1, y1, x2, y2):
    A = x - x1
    B = y - y1
    C = x2 - x1
    D = y2 - y1

    dot = A * C + B * D
    len_sq = C * C + D * D
    param = dot / len_sq if len_sq != 0 else -1

    if param < 0:
        xx = x1
        yy = y1
    elif param > 1:
        xx = x2
        yy = y2
    else:
        xx = x1 + param * C
        yy = y1 + param * D

    dx = x - xx
    dy = y - yy
    return np.sqrt(dx**2 + dy**2)


def pDistanceParallel(test_points, line_points):
    test_points, line_points = test_points.astype(float), line_points.astype(float)
    A = test_points[None, :, :] - line_points[:-1, None, :]
    B = line_points[1:, :] - line_points[:-1, :]

    dot = A[:, :, 0] * B[:, None, 0] + A[:, :, 1] * B[:, None, 1]
    len_sq = B[:, 0]**2 + B[:, 1]**2

    param = np.full_like(dot, -1)
    param[len_sq != 0.] = dot / len_sq[:, None]
    param = np.repeat(param[:, :, None], 2, axis = 2)

    closest = line_points[:-1, None, :] + param * B[:, None, :]
    np.putmask(closest, param < 0, line_points[:-1])
    np.putmask(closest, param > 1, line_points[1:])


    dist = np.linalg.norm(closest - test_points, axis = 2)
    return np.min(dist, axis = 0)

def DouglasPeucker(input, epsilon, depth = 0):
    if input.shape[0] < 3:
        return input
    distances = pDistanceParallel(input[:, :2], input[[0, -1], :2])
    distances[input[:, 2] == 1] = 0     # We can't chose a point that is invisible
    # distances[1:][input[:-1, 2] == 1] = 0     # We can't chose a point that is after invisible ??
    index = np.argmax(distances[1:-1]) + 1  # Index + 1 since we've sliced the first element away
    dmax = distances[index]

    Resultlist = []
    if dmax > epsilon:
        recResults1 = DouglasPeucker(input[:index+1], epsilon, depth = depth + 1)
        recResults2 = DouglasPeucker(input[index:], epsilon, depth = depth + 1)

        Resultlist = np.vstack([recResults1[:-1], recResults2])
    else:
        # We don't want to delete invisible line segments
        undeletable = (input[:, 2] == 1) | (np.concatenate([np.array([0]), input[:-1, 2]]) == 1)
        undeletable[[0, -1]] = 1
        Resultlist = input[undeletable]

    return Resultlist