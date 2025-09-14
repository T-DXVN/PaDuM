import torch.nn as nn
# from layers.net_CNN import MyCNN
# from layers.net_Mamba import MyMamba
from layers.network import DualStreamNet
from layers.revin import RevIN
from layers.decomp import DECOMP
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels
        d_state = configs.d_state
        d_model = configs.d_model
        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta   

        #layer
        #self.mycnn = MyCNN(seq_len, pred_len, patch_len, stride, padding_patch,1)
        #self.mymamba = MyMamba(seq_len, pred_len, patch_len, stride, padding_patch,d_state,d_model)
        self.mamba_cnn = DualStreamNet(seq_len, pred_len, patch_len, stride, padding_patch,1,d_state,d_model)
        self.decomp = DECOMP(self.ma_type, alpha, beta)
        

    def forward(self, x):
        # x: [Batch, Input, Channel]

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            #x = self.mycnn(x)
            #x = self.mymamba(x)
            x = self.mamba_cnn(x,x)
            
        else:
            seasonal_init,trend_init=self.decomp(x)
            x = self.mamba_cnn(seasonal_init,trend_init)
           

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x