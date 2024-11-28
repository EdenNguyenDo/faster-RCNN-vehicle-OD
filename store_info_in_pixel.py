from itertools import count
import cv2


img = cv2.imread("dataset/train/train_images/frame1.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rows, cols,_ = img.shape
print(count(img[0][1437]))
# for i in range(rows):
#     for j in range(cols):
#         k = img[i,j]
#         print(k)

