import torch.nn as nn
from src.tasks.crp_model import BertLayerNorm, GeLU

class MLP(nn.Module):
    def __init__(self,nb_class):
        super().__init__()

        self.logit_fc = nn.Sequential(
                    nn.Linear(33792, 1024),
                    BertLayerNorm(1024, eps=1e-12),
                    GeLU(),
                    nn.Linear(1024, nb_class)
                )

        self.dropout = nn.Dropout(0.5)
    def forward(self,x):
        # x = self.dropout(x)
        x = self.logit_fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.Generator_CNN = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.Generator_C3D = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )


        self.unmp2d = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.unmp3d = nn.MaxUnpool3d(kernel_size=[2,2,2],stride=[1,2,2])
        self.LD = nn.Linear(1536,12*7*16)
        self.LC = nn.Linear(1536,64*22)

        self.dropout = nn.Dropout(p=0.5)
        self.SmothL1loss = nn.SmoothL1Loss()

    def forward(self, x, indices_r, indices_v, radar_ori, video_ori):

        x = self.dropout(x)
        cnn_x = self.LC(x)
        c3d_x = self.LD(x)

        cnn_x = cnn_x.reshape((cnn_x.size()[0], 64, 22, 22))
        c3d_x = c3d_x.reshape((c3d_x.size()[0], 16, 22, 7, 12))
        cnn_x = self.unmp2d(cnn_x,indices_r)
        c3d_x = self.unmp3d(c3d_x,indices_v)
        Gene_cnn = self.Generator_CNN(cnn_x)
        Gene_c3d = self.Generator_C3D(c3d_x)
        LL =video_ori[:,:,:,:,0:47]
        loss_G_cnn = self.SmothL1loss(radar_ori, Gene_cnn)
        loss_G_c3d = self.SmothL1loss(LL, Gene_c3d)

        return loss_G_cnn,loss_G_c3d