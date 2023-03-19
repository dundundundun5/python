import imageio.v3
def transparence2white(img):
    sp = img.shape  # 获取图片维度
    width = sp[0]  # 宽度
    height = sp[1]  # 高度
    for yh in range(height):
        for xw in range(width):
            color_d = img[xw, yh]  # 遍历图像每一个点，获取到每个点4通道的颜色数据
            if (color_d[3] == 0):  # 最后一个通道为透明度，如果其值为0，即图像是透明
                img[xw, yh] = [255, 255, 255, 255]  # 则将当前点的颜色设置为白色，且图像设置为不透明
    return img
import cv2
fps = 2
loop = 1
png_list = []
png_files = [str(i) for i in range(1, 49, 1)]
path = "D:/gif/"
for png_name in png_files:
    png = cv2.imread(path + png_name + ".png", -1)
    png = transparence2white(png)
    # cv2.imshow("img",png)
    png_list.append(png)
#imageio.m
imageio.mimsave("permute2.gif", png_list, 'GIF', fps=2,loop=1)