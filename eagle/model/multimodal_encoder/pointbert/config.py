from transformers import PretrainedConfig

class PointTransformerConfig(PretrainedConfig):
    model_type = "pointbert"

    def __init__(self, 
                 trans_dim=384, 
                 depth=12, 
                 drop_path_rate=0.1, 
                 cls_dim=40, 
                 num_heads=6,
                 group_size=32, 
                 num_group=512,
                 encoder_dims=256,
                 point_dims=3,
                 use_max_pool=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.trans_dim = trans_dim
        self.depth = depth
        self.drop_path_rate = drop_path_rate
        self.cls_dim = cls_dim
        self.num_heads = num_heads
        self.group_size = group_size
        self.num_group = num_group
        self.encoder_dims = encoder_dims
        self.point_dims = point_dims
        self.use_max_pool = use_max_pool