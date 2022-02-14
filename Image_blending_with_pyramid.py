import cv2
import matplotlib.pyplot as plt
import numpy as np


# Create a display function using matplotlib
def display(images):
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
    plt.show()


# Read 2 images apple and orange
A = cv2.imread('images/Image blending with pyramid/apple.jpg')
A = cv2.resize(A, (500, 500))
A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
B = cv2.imread('images/Image blending with pyramid/orange.jpg')
B = cv2.resize(B, (500, 500))
B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)

# Generate Gaussian pyramid for A and display
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
display(gpA)

# generate Gaussian pyramid for B and display
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)
display(gpB)

# generate Laplacian Pyramid for A and display
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i], dstsize=gpA[i - 1].shape[:2])
    L = cv2.subtract(gpA[i - 1], GE)
    lpA.append(L)

display(lpA)

# generate Laplacian Pyramid for B and display
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i], dstsize=gpB[i - 1].shape[:2])
    L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

display(lpB)

# Now add left and right halves of images in each level and display
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
    LS.append(ls)
display(LS)

# now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_, dstsize=LS[i].shape[:2])
    ls_ = cv2.add(ls_, LS[i])
plt.imshow(ls_)
plt.show()
