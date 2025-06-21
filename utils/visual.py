import numpy as np
import matplotlib.pyplot as plt
from torch import Tensor
import torch

def imshow(input : Tensor,mean : list,std : list,title=None) -> None:
    #matplotlibで表示するため(H,W,C)に変換
    input = input.numpy().transpose((1,2,0))

    #画像を正規化の逆変換
    mean = np.array(mean)
    std = np.array(std)
    input = std * input + mean

    #値を0~1に収める
    input = np.clip(input,0,1)

    #画像を表示
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def visualize_model(model,dataloaders,device,class_names,num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs[1], 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j],mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

    