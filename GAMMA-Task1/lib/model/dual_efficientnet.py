import torch.nn as nn
import torch
import pdb
from efficientnet_pytorch import EfficientNet

class Dual_efficientnetb7(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb7
    """
    def __init__(self):
        super(Dual_efficientnetb7, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b7')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b7')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(2560 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 64, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit

class Dual_efficientnetb6(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb6
    """
    def __init__(self):
        super(Dual_efficientnetb6, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b6')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b6')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(2304 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 56, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit


class Dual_efficientnetb4(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb4
    """
    def __init__(self):
        super(Dual_efficientnetb4, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b4')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b4')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(1792 * 2, 3) # ResNet34 use basic block, expansion = 1


        self.oct_branch._conv_stem = nn.Conv2d(256, 48, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit
        # return b1, b2

class Dual_efficientnetb5(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb5
    """
    def __init__(self):
        super(Dual_efficientnetb5, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b5')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b5')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(2048 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 48, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit

class Dual_efficientnetb3(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb3
    """
    def __init__(self):
        super(Dual_efficientnetb3, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b3')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b3')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)


        # self._avg_pooling1 = nn.AdaptiveAvgPool2d(1)
        # self._avg_pooling2 = nn.AdaptiveAvgPool2d(1)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(1536 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 40, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit


class Dual_efficientnetb2(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb2
    """
    def __init__(self):
        super(Dual_efficientnetb2, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b2')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b2')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(1408 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 32, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit

class Dual_efficientnetb1(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb1
    """
    def __init__(self):
        super(Dual_efficientnetb1, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b1')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b1')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(1280 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 32, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit

class Dual_efficientnetb0(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single efficientnetb0
    """
    def __init__(self):
        super(Dual_efficientnetb0, self).__init__()


        self.fundus_branch = EfficientNet.from_pretrained('efficientnet-b0')
        self.oct_branch = EfficientNet.from_pretrained('efficientnet-b0')
        self._avg_pooling1 = self.fundus_branch._avg_pooling
        self._avg_pooling2 = self.oct_branch._avg_pooling
        self._dropout = nn.Dropout(0.5)

        # self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        # self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(1280 * 2, 3) # ResNet34 use basic block, expansion = 1



        self.oct_branch._conv_stem = nn.Conv2d(256, 32, kernel_size=3, stride=2, bias=False)





    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch.extract_features(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch.extract_features(oct_img)  # ([bs, 512])
        # print(b2.size())
        # print(b1.size())

        b1 = self._avg_pooling1(b1)
        b2 = self._avg_pooling2(b2)

        # print(b2.size())
        # print(b1.size())

        b1 = torch.flatten(b1, 1)
        b2 = torch.flatten(b2, 1)

        # print(b2.size())
        # print(b1.size())

        b12 = torch.cat([b1, b2], 1)
        b12 = self._dropout(b12)

        logit = self.decision_branch(b12)

        return logit




if __name__ == "__main__":


    model = Dual_efficientnetb3()
    pdb.set_trace()
    # model = EfficientNet.from_pretrained('efficientnet-b4')

    # print(model)
    inputs1 = torch.rand(4,3,512,512)
    inputs2 = torch.rand(4,256,512,512)
    # x = model.extract_features(inputs)
    # pdb.set_trace()
    # print(x)

    output = model(inputs1, inputs2)

    pdb.set_trace()
    print(output)

    # inputs = torch.rand(1, 3, 512, 512)
    # model = EfficientNet.from_pretrained('efficientnet-b4')
    # endpoints = model.extract_endpoints(inputs)
    # pdb.set_trace()
    # print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
    # print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
    # print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
    # print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
    # print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
    # print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])