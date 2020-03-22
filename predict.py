import torch
import cv2
import torch.nn.functional as F
from torchvision import datasets, transforms
from lenet_5 import LeNet
from PIL import Image


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式

    img_ori = cv2.imread("n.jpg")  # 读取要预测的图片
    # width, height = img_ori.shape[:2][::-1]
    # img_resize = cv2.resize(img_ori,
    #                         (int(width * 0.5), int(height * 0.5)), interpolation=cv2.INTER_CUBIC)
    # mg = Image.fromarray(cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB))
    img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

    trans = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((.5,.5,.5),(.5,.5,.5)),
    ])
    PIL_image = Image.fromarray(img_gray)  # 这里ndarray_image为原来的numpy数组类型的输入
    img = trans(PIL_image)
    img = img.to(device)
    img = img.unsqueeze(1)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    output = model(img)
    prob = F.softmax(output, dim=1)  # prob是2个分类的概率
    print(prob)
    value, predicted = torch.max(output.data, 1)
    print(predicted.item())
    print(value)
