import cv2

path = "/Users/will/projects/legoproj/data/pole_single/1472_pole_a.png" 


img = cv2.imread( path,0 )#[...,::-1]


x1 = 294
y1 = 387

x2 = 324
y2 = 348

'''
x1 = 77
y1 = 104

x2 = 97
y2 = 76
'''

cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), 2)

cv2.imshow('tst',img)
#cv2.waitkey(0)
cv2.waitKey(0)

print(tuple([x1, y1, x2 - x1, y2 - y1]))