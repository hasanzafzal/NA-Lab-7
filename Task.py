
import numpy as np
import matplotlib.pyplot as plt
from math import factorial


# Given data points for the left half of the face.
y = np.array([0, 2, 4, 6, 8, 10], dtype=float)
x = np.array([0.0, 1.1, 1.8, 2.4, 2.0, 1.3], dtype=float)

h = y[1] - y[0]
n = len(x)

# Construct the forward difference table.
diff = [x.copy()]

while len(diff[-1]) > 1:
    diff.append(np.diff(diff[-1]))


# Newton's forward interpolation formula.
def forward(value):
    origin = min(int((value - y[0]) / h), n - 1)
    p = (value - y[origin]) / h
    result = x[origin]
    product = 1.0

    for k in range(1, n - origin):
        product *= p - k + 1
        result += product * diff[k][origin] / factorial(k)

    return result


# Newton's backward interpolation formula.
def backward(value):
    origin = min(int(np.ceil((value - y[0]) / h)), n - 1)
    p = (value - y[origin]) / h
    result = x[origin]
    product = 1.0

    for k in range(1, origin + 1):
        product *= p + k - 1
        result += product * diff[k][origin - k] / factorial(k)

    return result


# Stirling's central difference interpolation formula.
def stirling(value):
    origin = int(round((value - y[0]) / h))
    origin = max(1, min(origin, n - 2))

    p = (value - y[origin]) / h
    result = x[origin]

    for order in range(1, n):
        if order % 2 == 1:
            r = (order - 1) // 2
            left = origin - r - 1
            right = origin - r

            if left < 0 or right + order > n - 1:
                break

            product = p
            for j in range(1, r + 1):
                product *= p * p - j * j

            average = (diff[order][left] + diff[order][right]) / 2
            result += product * average / factorial(order)

        else:
            r = order // 2
            index = origin - r

            if index < 0 or index + order >= n:
                break

            product = p * p
            for j in range(1, r):
                product *= p * p - j * j

            result += product * diff[order][index] / factorial(order)

    return result


# Generate intermediate vertical coordinates.
y_left = np.linspace(y[0], y[-1], 201)
x_left = []

# Apply the appropriate interpolation formula.
for value in y_left:
    if value < 4:
        x_left.append(forward(value))
    elif value > 6:
        x_left.append(backward(value))
    else:
        x_left.append(stirling(value))

x_left = np.array(x_left)

# Mirror the left outline across the vertical midline.
x_right = -x_left

# Join both halves to form one complete outline.
x_full = np.concatenate((x_left[::-1], x_right[1:]))
y_full = np.concatenate((y_left[::-1], y_left[1:]))

# Plot the complete face outline.
plt.figure(figsize=(7, 8))
plt.plot(x_full, y_full, label="Interpolated face outline")
plt.scatter(-x, y, color="red", label="Left data points")
plt.scatter(x[1:], y[1:], color="green", label="Mirrored data points")

plt.axvline(0, color="gray", linestyle="--", label="Vertical midline")
plt.xlabel("Horizontal distance (x)")
plt.ylabel("Vertical position (y)")
plt.title("Complete Symmetrical Face Outline")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.show()
