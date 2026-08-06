import torch
import torch.nn as nn

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


    def forward(self,x):
        return self.conv(x)

#Down class
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels,out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2 , stride=2)

    def forward(self,x):
        skip = self.conv(x)
        x = self.pool(skip)

        return skip,x

#Bottleneck class

class Bottleneck(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self,x):
        return self.conv(x)


#Up class

class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels*2, out_channels)

    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([skip, x],dim=1)
        x = self.conv(x)

        return x
    

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        #Encoder
        self.down1 = Down(4,64)
        self.down2 = Down(64,128)
        self.down3 = Down(128,256)
        self.down4 = Down(256,512)

        #Bottleneck
        self.bottleneck = Bottleneck(512,1024)

        #Decoder
        self.up1 = Up(1024,512)
        self.up2 = Up(512,256)
        self.up3 = Up(256,128)
        self.up4 = Up(128,64)

        #Final Output layer
        self.final_conv = nn.Conv2d(64 , 1, kernel_size=1)

    def forward(self, x):
            skip1, x = self.down1(x)
            skip2, x = self.down2(x)
            skip3, x = self.down3(x)
            skip4, x = self.down4(x)
        
            x = self.bottleneck(x)
        
            x = self.up1(x, skip4)
            x = self.up2(x, skip3)
            x = self.up3(x, skip2)
            x = self.up4(x, skip1)
        
            x = self.final_conv(x)
        
            return x

 
    

def main():
    pass

    

if __name__ == "__main__":
    main()