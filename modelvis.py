import os
import sys
import argparse
import numpy as np
import torch
import torchvision
import inception
import inceptionv4
import densenet
from torch import optim
from torch.autograd import Variable
from scipy.misc import imsave

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    #x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def removeParallel(model):
    while isinstance(model, torch.nn.parallel.DataParallel):
        model = model.module
    return model

def updateInput(grad):
    global randInput,randInputData
    normGrad = grad / (torch.sqrt(torch.mean(torch.mul(grad,grad))) + 1e-5) * 200
    randInput = randInput + normGrad
    randInputData = randInput.data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""Generate image presentation of model.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--modelPath', default=None, type=str, help='Path to model that will be visualized')
    parser.add_argument('--outputPath', default=None, type=str, help='Path to store the output image')
    parser.add_argument('--imgSize', default=299, type=int, help='size of the image that network accept')
    parser.add_argument('--preview', default=999999, type=int, help='number of filter per filter group for previewing')
    parser.add_argument('--selectedFilterGroup', default=None, type=str, help='selected filter group to preview, ex. Conv2d_1a_3x3,Conv2d_2a_3x3')
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args()
    
    
    outputPath = args.outputPath
    modelPath = args.modelPath
    imgSize = args.imgSize
    imgPerFilterGroup = args.preview
    
    randInput1 = np.random.random((2,3,imgSize,imgSize)) * 20 + 128
    
    randInput = Variable(torch.FloatTensor(randInput1),requires_grad=True).cuda()
    randInputData = randInput.data
    filterGroup = None
    classes = 0
    
    if modelPath is None:
        model = inception.inception_v3(pretrained=True)
    else:
        model = torch.load(modelPath)
        model=removeParallel(model)
        o1,filters = model(randInput)
        classes = o1.data.cpu().numpy().shape[-1]
        print('classes',classes)
        
    model.setVisMode(True)
    model.cuda()
    o1,filters = model(randInput)
    for fg in filters.keys():
        filterGroup = fg
        if args.selectedFilterGroup is not None:
            #we skip the filter group which is not interested
            if fg not in args.selectedFilterGroup:
                continue
        
        for f in range(min(filters[fg].shape[1],imgPerFilterGroup)):
            print('visualizing filter group ', fg, 'filter No.', f)
            randInput1 = np.random.random((2,3,imgSize,imgSize)) * 20 + 128
            randInput = Variable(torch.FloatTensor(randInput1),requires_grad=True).cuda()
            randInputData = randInput.data
            optimizer = optim.SGD(model.parameters(),lr=0, momentum=0.9)
            randInput.register_hook(updateInput)
            skip = False
            for i in range(30):
                optimizer.zero_grad()
                randInput = Variable(torch.FloatTensor(randInput1),requires_grad=True).cuda()
                randInput.data = randInputData
                randInput.register_hook(updateInput)
                o1,visDict = model(randInput)
                loss = visDict[fg][:,f].mean()
                if loss.data.cpu().numpy() == 0:
                    skip = True
                    break
                loss.backward()
            if skip == True:
                print('skipping {} filter no. {}'.format(filterGroup, f))
                continue
            img = randInputData.cpu().numpy()[0]
            img = deprocess_image(img)
            imgFolder = os.path.join(outputPath,filterGroup)
            if not os.path.exists(imgFolder):
                os.makedirs(imgFolder)
            imsave(os.path.join(imgFolder,'{}_{}.png'.format(filterGroup,f)), img)
    
