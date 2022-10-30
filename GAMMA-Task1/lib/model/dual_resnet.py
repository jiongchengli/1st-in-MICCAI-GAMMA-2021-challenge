import torch.nn as nn
import torchvision.models as models
import torch
import pdb

class Dual_network34(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet34
    """
    def __init__(self):
        super(Dual_network34, self).__init__()

        # resnet34 = models.resnet34()
        # self.fundus_branch = nn.Sequential(*list(resnet34.modules())[:-1])  # remove final fc
        # self.oct_branch = nn.Sequential(*list(resnet34.modules())[:-1])  # remove final fc

        self.fundus_branch = models.resnet34(pretrained=True) # remove final fc
        self.oct_branch = models.resnet34(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(512 * 1 * 2, 3) # ResNet34 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2d(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          bias=False)  # bias_attr

        self.oct_branch.fc = nn.Sequential()  # remove fc
        self.fundus_branch.fc = nn.Sequential()



    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch(oct_img)  # ([bs, 512])
        # print('------')
        # print(b2.size())
        # print(b1.size())
        # b1 = paddle.flatten(b1, 1)
        # b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(torch.cat([b1, b2], 1))

        return logit
        # return b1, b2



class Dual_network18(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnet18
    """
    def __init__(self):
        super(Dual_network18, self).__init__()

        # resnet34 = models.resnet34()
        # self.fundus_branch = nn.Sequential(*list(resnet34.modules())[:-1])  # remove final fc
        # self.oct_branch = nn.Sequential(*list(resnet34.modules())[:-1])  # remove final fc

        self.fundus_branch = models.resnet18(pretrained=True) # remove final fc
        self.oct_branch = models.resnet18(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(512 * 1 * 2, 3) # ResNet34 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2d(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          bias=False)  # bias_attr

        self.oct_branch.fc = nn.Sequential()  # remove fc
        self.fundus_branch.fc = nn.Sequential()



    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch(oct_img)  # ([bs, 512])
        # print('------')
        # print(b2.size())
        # print(b1.size())
        # b1 = paddle.flatten(b1, 1)
        # b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(torch.cat([b1, b2], 1))

        return logit

class Dual_networkx50(nn.Module):
    """
    simply create a 2-branch network, and concat global pooled feature vector.
    each branch = single resnetx50
    """
    def __init__(self):
        super(Dual_networkx50, self).__init__()

        # resnet34 = models.resnet34()
        # self.fundus_branch = nn.Sequential(*list(resnet34.modules())[:-1])  # remove final fc
        # self.oct_branch = nn.Sequential(*list(resnet34.modules())[:-1])  # remove final fc

        self.fundus_branch = models.resnext50_32x4d(pretrained=True) # remove final fc
        self.oct_branch = models.resnext50_32x4d(pretrained=True) # remove final fc
        self.decision_branch = nn.Linear(2048 * 1 * 2, 3) # ResNet34 use basic block, expansion = 1

        # replace first conv layer in oct_branch
        self.oct_branch.conv1 = nn.Conv2d(256, 64,
                                          kernel_size=7,
                                          stride=2,
                                          padding=3,
                                          bias=False)  # bias_attr

        self.oct_branch.fc = nn.Sequential()  # remove fc
        self.fundus_branch.fc = nn.Sequential()



    def forward(self, fundus_img, oct_img):
        b1 = self.fundus_branch(fundus_img)
        # pdb.set_trace()
        b2 = self.oct_branch(oct_img)  # ([bs, 512])
        # print('------')
        # print(b2.size())
        # print(b1.size())
        # b1 = paddle.flatten(b1, 1)
        # b2 = paddle.flatten(b2, 1)
        logit = self.decision_branch(torch.cat([b1, b2], 1))

        return logit