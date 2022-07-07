#创建颜色字典
from PIL import Image
import cv2
import os
import pathlib
import numpy as np
from tqdm import tqdm

def color_dict(labelFolder):
    colorDict = np.array([])
    #  获取文件夹内的所有文件名
    ImageNameList = os.listdir(labelFolder)  # ['105.tif', '20.tif', '77.tif',... '116.tif']
    for i in tqdm(range(len(ImageNameList))):
        ImagePath = labelFolder + "/" + ImageNameList[i]
        image = np.array(Image.open(ImagePath))
        color = np.unique(image)
        colorDict = np.append(color, colorDict)
    print(np.unique(colorDict))


if __name__ == "__main__":
    color_dict(r"D:\MyLab\datasets\VOC\masks\train2012")