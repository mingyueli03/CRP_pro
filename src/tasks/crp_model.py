import math
import sys
import torch
from torch import nn
from src.param import args

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

BertLayerNorm = torch.nn.LayerNorm


class BertAttention(nn.Module):
    def __init__(self, config, ctx_dim=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim =config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)

        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertAttOutput(nn.Module):
    def __init__(self, config):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor):
        output = self.att(input_tensor, ctx_tensor)
        attention_output = self.output(output, input_tensor)
        return attention_output


class BertSelfattLayer(nn.Module):
    def __init__(self, config):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(config) #attention
        self.output = BertAttOutput(config)

    def forward(self, input_tensor):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class PositionalEncoding_v(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config):
        super(PositionalEncoding_v, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 22, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class PositionalEncoding_r(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config):
        super(PositionalEncoding_r, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, 22, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

"""
---------------------------------------------------------------------------------------
      Above modules are copied from BERT (pytorch-transformer) with modifications.
---------------------------------------------------------------------------------------
"""


class CRPLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.radar_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.radar_inter = BertIntermediate(config)
        self.radar_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, radar_input, visn_input):
        radar_att_output = self.visual_attention(radar_input, visn_input)
        visn_att_output = self.visual_attention(visn_input, radar_input)
        return radar_att_output, visn_att_output

    def self_att(self, radar_input, visn_input):
        radar_att_output = self.radar_self_att(radar_input)
        visn_att_output = self.visn_self_att(visn_input)
        return radar_att_output, visn_att_output

    def output_fc(self, radar_input, visn_input):
        # FC layers
        radar_inter_output = self.radar_inter(radar_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.radar_output(radar_inter_output, radar_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, radar_feats,visn_feats):
        radar_att_output = radar_feats
        visn_att_output = visn_feats

        radar_att_output, visn_att_output = self.cross_att(radar_att_output,visn_att_output)
        radar_att_output, visn_att_output = self.self_att(radar_att_output,visn_att_output)
        radar_output, visn_output = self.output_fc(radar_att_output, visn_att_output)

        return radar_output, visn_output


class CRPEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        print("crp encoder with %d r_layers, %d x_layers, and %d v_layers." %
              (args.r_layers, args.x_layers, args.v_layers))

        # Layers
        self.rlayers = nn.ModuleList(
            [BertLayer(args) for _ in range(args.r_layers)]
        )
        self.xlayers = nn.ModuleList(
            [CRPLayer(args) for _ in range(args.x_layers)]
        )
        self.vlayers = nn.ModuleList(
            [BertLayer(args) for _ in range(args.v_layers)]
        )
        self.PositionalEn_r = PositionalEncoding_r(args)
        self.PositionalEn_v = PositionalEncoding_v(args)

    def forward(self, radar_feature_output,video_feature_output):

        radar_feats = self.PositionalEn_r(radar_feature_output)
        visn_feats = self.PositionalEn_v(video_feature_output)

        # Run radar layers
        for layer_module in self.rlayers:
            radar_feats = layer_module(radar_feats)
        # Run video layers
        for layer_module in self.vlayers:
            visn_feats = layer_module(visn_feats)

        # Run cross-modality layers
        for layer_module in self.xlayers:
            radar_feats, visn_feats = layer_module(radar_feats,visn_feats)

        return radar_feats, visn_feats

class CRPModel(nn.Module):
    def __init__(self,nb_class):
        super().__init__()
        self.encoder = CRPEncoder(args)

        self.conv_1 = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),  # input[1,3,16,128,171]
            nn.BatchNorm3d(8),
            nn.ReLU(),

        )
        self.conv_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(),

        )

        self.cnn1 = nn.Sequential(  # 模型会依次执行Sequential中的函数
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),  # output: 64 * 128 * 128
            nn.BatchNorm2d(16),
            nn.ReLU(),

        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),  # output: 128 * 64 * 64
            nn.BatchNorm2d(32),
            nn.ReLU(),

        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),  # output: 256 * 32 * 32
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.fc_cnn = nn.Sequential(
            nn.Linear(64 * 22, 768),
            BertLayerNorm(768, eps=1e-12),
            GeLU()
        )
        self.fc_c3d = nn.Sequential(
            nn.Linear(1344, 768),
            BertLayerNorm(768, eps=1e-12),
            GeLU()
        )

        self.logit_fc = nn.Sequential(
            # fusion
            nn.Linear(33792, 1024),
            BertLayerNorm(1024, eps=1e-12),
            GeLU(),
            nn.Linear(1024, nb_class)
        )


        self.dropout = nn.Dropout(p=0.5)
        self.maxp2d = nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, return_indices = True)
        self.maxp3d = nn.MaxPool3d(kernel_size = [2, 2, 2],stride = [1, 2, 2], return_indices = True)
    def forward(self, radar_feature_output, video_feature_output):

        radar_x = self.cnn1(radar_feature_output)
        radar_x,indices_r = self.maxp2d(radar_x)
        radar_x = self.cnn2(radar_x)
        radar_ori,indices_rori = self.maxp2d(radar_x)
        radar_x = self.cnn3(radar_ori)
        radar_x,indices_r = self.maxp2d(radar_x)

        radar_feature = radar_x.view(radar_x.size()[0], radar_x.size()[2], -1)
        radar_feature = self.dropout(radar_feature)
        radar_feature = self.fc_cnn(radar_feature)

        video_feature = self.conv_1(video_feature_output)
        video_ori,indices_vori = self.maxp3d(video_feature)
        video_x = self.conv_2(video_ori)
        video_x,indices_v = self.maxp3d(video_x)
        video_feature = video_x.view(video_x.size()[0], video_x.size()[2], -1)
        video_feature = self.dropout(video_feature)
        video_feature = self.fc_c3d(video_feature)

        x_r,x_v = self.encoder(radar_feature, video_feature)

        return x_r,x_v,indices_r, indices_v, radar_ori, video_ori

