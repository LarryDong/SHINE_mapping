import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import grad

from utils.config import SHINEConfig


class Decoder(nn.Module):
    def __init__(self, config: SHINEConfig, is_geo_encoder = True, is_time_conditioned = False): 
        # 这个decoder貌似还预测了语义标签
        super().__init__()          # 调用父类的初始化方法
        
        if is_geo_encoder:
            mlp_hidden_dim = config.geo_mlp_hidden_dim
            mlp_bias_on = config.geo_mlp_bias_on
            mlp_level = config.geo_mlp_level
        else:
            mlp_hidden_dim = config.sem_mlp_hidden_dim
            mlp_bias_on = config.sem_mlp_bias_on
            mlp_level = config.sem_mlp_level

        input_layer_count = config.feature_dim
        if is_time_conditioned:
            input_layer_count += 1

        # predict sdf (now it anyway only predict sdf without further sigmoid
        # Initializa the structure of shared MLP
        layers = []
        for i in range(mlp_level):          # 添加隐藏层hidden layer，论文/代码中使用2层。  输入L=8 -> HiddenLayer1=32 -> 
            if i == 0:
                layers.append(nn.Linear(input_layer_count, mlp_hidden_dim, mlp_bias_on))    # 输入和hiddenlayer1之间是，L=8 连接 dim=32
            else:
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim, mlp_bias_on))       # hiddenlayer之间，是 dim=32 连接 dim=32
        self.layers = nn.ModuleList(layers)
        self.lout = nn.Linear(mlp_hidden_dim, 1, mlp_bias_on)                               # hiddenlayer最后一层(2)，和输出层，是 dim=32 -> 1
        self.nclass_out = nn.Linear(mlp_hidden_dim, config.sem_class_count + 1, mlp_bias_on) # sem class + free space class
        # self.bn = nn.BatchNorm1d(self.hidden_dim, affine=False)

        self.to(config.device)
        # torch.cuda.empty_cache()

    def forward(self, feature):
        # If we use BCEwithLogits loss, do not need to do sigmoid mannually
        output = self.sdf(feature)
        return output

    # predict the sdf (opposite sign to the actual sdf)
    def sdf(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out
    
    def time_conditionded_sdf(self, sum_features, ts):      # 预测“时间条件下的SDF”值
        time_conditioned_feature = torch.torch.cat((sum_features, ts.view(-1, 1)), dim=1)

        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(time_conditioned_feature))
            else:
                h = F.relu(l(h))

        out = self.lout(h).squeeze(1)
        # linear (feature_dim + 1 -> hidden_dim)
        # relu
        # linear (hidden_dim -> hidden_dim)
        # relu
        # linear (hidden_dim -> 1)

        return out

    # predict the occupancy probability
    def occupancy(self, sum_features):
        out = torch.sigmoid(self.sdf(sum_features))  # to [0, 1]
        return out

    # predict the probabilty of each semantic label
    def sem_label_prob(self, sum_features):
        for k, l in enumerate(self.layers):
            if k == 0:
                h = F.relu(l(sum_features))
            else:
                h = F.relu(l(h))

        out = F.log_softmax(self.nclass_out(h), dim=1)
        return out

    def sem_label(self, sum_features):
        out = torch.argmax(self.sem_label_prob(sum_features), dim=1)
        return out
