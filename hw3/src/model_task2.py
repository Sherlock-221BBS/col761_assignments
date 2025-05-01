import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroBaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.hidden_channels=hidden_channels; self.out_channels=out_channels; self.num_layers=num_layers
        self.convs=torch.nn.ModuleList(); self.lin=Linear(hidden_channels,out_channels)
    def _build_conv_layer(self,is_last=False): raise NotImplementedError
    def __init_layers__(self): [self.convs.append(self._build_conv_layer(is_last=(i==self.num_layers-1))) for i in range(self.num_layers)]
    def forward(self, x_dict, edge_index_dict):
        intermediate_x_dict = x_dict
        for conv in self.convs: intermediate_x_dict = conv(intermediate_x_dict,edge_index_dict); intermediate_x_dict={k:x.relu() for k,x in intermediate_x_dict.items()}
        return self.lin(intermediate_x_dict['user'])

class HeteroGraphSAGE(HeteroBaseGNN):
     def __init__(self, hidden_channels, out_channels, num_layers=2):
          super().__init__(hidden_channels=hidden_channels, out_channels=out_channels, num_layers=num_layers)
          self.__init_layers__()

     def _build_conv_layer(self, is_last=False):
          conv_dict = {
               ('user', 'interacts_with', 'product'): SAGEConv((-1, -1), self.hidden_channels, aggr='mean'),
               ('product', 'rev_interacts_with', 'user'): SAGEConv((-1, -1), self.hidden_channels, aggr='mean'),
          }
          return HeteroConv(conv_dict, aggr='sum')
