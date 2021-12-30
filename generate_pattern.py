import numpy as np
import os, cv2, math
from itertools import permutations

# Parameters
pattern_size = 100
black_line = False
colors = [
    [31, 31, 31],
    [31, 31, 223],
    [31, 127, 127],
    [31, 223, 31],
    [31, 223, 223],
    [127, 31, 31],
    [127, 31, 223],
    [127, 127, 127],
    [127, 223, 31],
    [127, 223, 223],
    [223, 31, 31],
    [223, 31, 223],
    [223, 127, 127],
    [223, 223, 31],
    [223, 223, 223]
]
color_num = len(colors)
colors = np.array(colors)
color_permutations = list(permutations(list(range(color_num)), 2))
pattern_count = 0
circle_r = pattern_size / 2
pattern_path = './patterns/'
if not os.path.exists(pattern_path):
    os.mkdir(pattern_path)

def diff_r(i, j, x, y):
    return math.sqrt(math.pow(i - x, 2) + math.pow(j - y, 2)) - circle_r


# Shape 1
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if (i + j) < pattern_size:
                img[i][j] = colors[permutation[0]]
            elif (i + j) > pattern_size:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 2
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if (i - j) > 0:
                img[i][j] = colors[permutation[0]]
            elif (i - j) < 0:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 3
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if ((i - j) > 0 and (i + j) < pattern_size) or ((i - j) < 0 and (i + j) > pattern_size):
                img[i][j] = colors[permutation[0]]
            elif ((i - j) < 0 and (i + j) < pattern_size) or ((i - j) > 0 and (i + j) > pattern_size):
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 4
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if (2 * i + j < pattern_size) or (2 * i - j > pattern_size):
                img[i][j] = colors[permutation[0]]
            elif (2 * i + j > pattern_size) and (2 * i - j < pattern_size):
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 5
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if (2 * i - j < 0) or (i + 0.5 * j > pattern_size):
                img[i][j] = colors[permutation[0]]
            elif (2 * i - j > 0) and (i + 0.5 * j < pattern_size):
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 6
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if (i - 2 * j > 0) or (0.5 * i + j > pattern_size):
                img[i][j] = colors[permutation[0]]
            elif (i - 2 * j < 0) and (0.5 * i + j < pattern_size):
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 7
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if (i + 2 * j < pattern_size) or (-i + 2 * j > pattern_size):
                img[i][j] = colors[permutation[0]]
            elif (i + 2 * j > pattern_size) and (-i + 2 * j < pattern_size):
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    img[i][j] = (colors[permutation[0]] + colors[permutation[1]]) / 2
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 8
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, circle_r, circle_r) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, circle_r, circle_r) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = diff_r(i, j, circle_r, circle_r) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 9
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, circle_r) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, circle_r) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = diff_r(i, j, 0, circle_r) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 10
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, circle_r, 0) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, circle_r, 0) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = diff_r(i, j, circle_r, 0) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 11
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, pattern_size, circle_r) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, pattern_size, circle_r) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = diff_r(i, j, pattern_size, circle_r) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 12
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, circle_r, pattern_size) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, circle_r, pattern_size) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = diff_r(i, j, circle_r, pattern_size) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 13
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, circle_r) <= -0.5 or diff_r(i, j, pattern_size, circle_r) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, circle_r) >= 0.5 and diff_r(i, j, pattern_size, circle_r) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, circle_r), diff_r(i, j, pattern_size, circle_r)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 14
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, circle_r, 0) <= -0.5 or diff_r(i, j, circle_r, pattern_size) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, circle_r, 0) >= 0.5 and diff_r(i, j, circle_r, pattern_size) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, circle_r, 0), diff_r(i, j, circle_r, pattern_size)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 15
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, 0) <= -0.5 or diff_r(i, j, 0, pattern_size) <= -0.5 or diff_r(i, j, pattern_size, pattern_size) <= -0.5 or diff_r(i, j, pattern_size, 0) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, 0) >= 0.5 and diff_r(i, j, 0, pattern_size) >= 0.5 and diff_r(i, j, pattern_size, pattern_size) >= 0.5 and diff_r(i, j, pattern_size, 0) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, 0), diff_r(i, j, 0, pattern_size), diff_r(i, j, pattern_size, pattern_size), diff_r(i, j, pattern_size, 0)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 16
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, 0) <= -0.5 or diff_r(i, j, pattern_size, pattern_size) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, 0) >= 0.5 and diff_r(i, j, pattern_size, pattern_size) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, 0), diff_r(i, j, pattern_size, pattern_size)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 17
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, pattern_size) <= -0.5 or diff_r(i, j, pattern_size, 0) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, pattern_size) >= 0.5 and diff_r(i, j, pattern_size, 0) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, pattern_size), diff_r(i, j, pattern_size, 0)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 18
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, 0) <= -0.5 or diff_r(i, j, pattern_size, 0) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, 0) >= 0.5 and diff_r(i, j, pattern_size, 0) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, 0), diff_r(i, j, pattern_size, 0)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 19
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, 0) <= -0.5 or diff_r(i, j, 0, pattern_size) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, 0) >= 0.5 and diff_r(i, j, 0, pattern_size) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, 0), diff_r(i, j, 0, pattern_size)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 20
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, pattern_size, 0) <= -0.5 or diff_r(i, j, pattern_size, pattern_size) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, pattern_size, 0) >= 0.5 and diff_r(i, j, pattern_size, pattern_size) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, pattern_size, 0), diff_r(i, j, pattern_size, pattern_size)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1

# Shape 21
for permutation in color_permutations:
    img = np.zeros((pattern_size, pattern_size, 3), np.uint8)
    for i in range(pattern_size):
        for j in range(pattern_size):
            if diff_r(i, j, 0, pattern_size) <= -0.5 or diff_r(i, j, pattern_size, pattern_size) <= -0.5:
                img[i][j] = colors[permutation[0]]
            elif diff_r(i, j, 0, pattern_size) >= 0.5 and diff_r(i, j, pattern_size, pattern_size) >= 0.5:
                img[i][j] = colors[permutation[1]]
            else:
                if not black_line:
                    alpha = min(diff_r(i, j, 0, pattern_size), diff_r(i, j, pattern_size, pattern_size)) + 0.5
                    img[i][j] = colors[permutation[0]] * (1 - alpha) + colors[permutation[1]] * alpha
    cv2.imwrite(pattern_path + str(pattern_count) + '.png', img)
    pattern_count += 1
