from torch import nn
class DownsampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(DownsampleLayer, self).__init__()
        self.Conv_BN_ReLU_2=nn.Sequential(
            nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.downsample(self.Conv_BN_ReLU_2(x))
class UpSampleLayer(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(UpSampleLayer, self).__init__()
        midchannel = 2
        self.Conv_BN_ReLU_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch * midchannel, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch * midchannel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch * midchannel, out_channels=out_ch * midchannel, kernel_size=3, stride=1,padding=1),
            nn.BatchNorm2d(out_ch * midchannel),
            nn.LeakyReLU(inplace=True)
        )
        self.upsample=nn.Sequential(
            nn.ConvTranspose2d(in_channels=out_ch * midchannel,out_channels=out_ch,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self,x):
        return self.upsample(self.Conv_BN_ReLU_2(x))

class Deep(nn.Module):
    def __init__(self, channels=1, level=7):
        super(Deep, self).__init__()
        self.nlevel = level
        out_channels=[channels*2**(i+4) for i in range(self.nlevel+1)]
        out_c = 8
        self.d = []
        self.u = []
        d = []
        u = []
        d.append(DownsampleLayer(channels, out_channels[0]).cuda())
        for x in range(self.nlevel):
            d.append(DownsampleLayer(out_channels[x],out_channels[x+1]).cuda())
        self.d = nn.ModuleList(d)
        for x in range(self.nlevel):
            u.append(UpSampleLayer(out_channels[self.nlevel-x],out_channels[self.nlevel-x-1]).cuda())
        u.append(UpSampleLayer(out_channels[0],out_c).cuda())
        self.u = nn.ModuleList(u)
        self.o=nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_c,1,3,1,1),
            nn.Sigmoid(),
        )
    def forward(self,out):
        for x in range(self.nlevel+1):
            out = self.d[x](out)
        for x in range(self.nlevel+1):
            out = self.u[x](out)
        return self.o(out)

