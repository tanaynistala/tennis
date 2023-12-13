import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

class FlowNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3),
            nn.LeakyReLU(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 2, 2),
            nn.LeakyReLU(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 5, 2, 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, 3, 1, 1),
            nn.LeakyReLU(0.1),
        )

        self.predict_flow6 = nn.Conv2d(1024, 2, 3, 1, 1)
        self.deconv5 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
        self.relu11 = nn.LeakyReLU(0.1)
        self.flow6_5 = nn.Conv2d(2, 512, 3, 2, 1, bias=False)

        self.predict_flow5 = nn.Conv2d(1026, 2, 3, 1, 1)
        self.deconv4 = nn.ConvTranspose2d(1026, 256, 4, 2, 1)
        self.relu12 = nn.LeakyReLU(0.1)
        self.flow5_4 = nn.Conv2d(514, 2, 4, 2, 1, bias=False)

        self.predict_flow4 = nn.Conv2d(514, 2, 3, 1, 1)
        self.deconv3 = nn.ConvTranspose2d(514, 128, 4, 2, 1)
        self.relu13 = nn.LeakyReLU(0.1)
        self.flow4_3 = nn.Conv2d(258, 2, 4, 2, 1, bias=False)

        self.predict_flow3 = nn.Conv2d(258, 2, 3, 1, 1)
        self.deconv2 = nn.ConvTranspose2d(258, 64, 4, 2, 1)
        self.relu14 = nn.LeakyReLU(0.1)
        self.flow3_2 = nn.Conv2d(130, 2, 4, 2, 1, bias=False)
        
    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)

        flow6 = self.predict_flow6(out_conv6)
        print(out_conv6.shape, flow6.shape)
        flow6_up = self.flow6_5(flow6)
        print(flow6_up.shape)
        out_deconv5 = self.relu11(self.deconv5(out_conv6))
        print(out_deconv5.shape)

        flow5 = self.predict_flow5(torch.cat([out_deconv5, flow6_up], dim=0))
        flow5_up = self.flow5_4(flow5)
        out_deconv4 = self.relu12(self.deconv4(torch.cat([out_deconv5, flow5_up], dim=1)))

        print(out_deconv4.shape, flow5_up.shape)
        flow4 = self.predict_flow4(torch.cat([out_deconv4, flow5_up], dim=1))
        flow4_up = self.flow4_3(flow4)
        out_deconv3 = self.relu13(self.deconv3(torch.cat([out_deconv4, flow4_up], dim=1)))

        flow3 = self.predict_flow3(torch.cat([out_deconv3, flow4_up], dim=1))
        flow3_up = self.flow3_2(flow3)
        out_deconv2 = self.relu14(self.deconv2(torch.cat([out_deconv3, flow3_up], dim=1)))

        flow2 = F.interpolate(out_deconv2, scale_factor=4, mode='bilinear', align_corners=False)

        if autograd.is_training():
            return flow2, flow3, flow4, flow5, flow6
        return flow2
