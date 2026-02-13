#多線程模組 - init_model 函數

from ultralytics import YOLO

def init_model():
    #在每個處理程序中初始化模型
    model = YOLO('new_10.pt')
    model.to('cuda')
    return model