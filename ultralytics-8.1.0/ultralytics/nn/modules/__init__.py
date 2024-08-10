# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxsim {f} {f} && open {f}')
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C3Ghost,
    C3x,
    GhostBottleneck,
    HGBlock,
    HGStem,
    Proto,
    RepC3,
    ResNetLayer,
    BasicStage, PatchEmbed_FasterNet, PatchMerging_FasterNet,

    SimAM, ECA, SpatialGroupEnhance, TripletAttention, CoordAtt, GAMAttention,
    SE, ShuffleAttention, SKAttention, DoubleAttention, CoTAttention, EffectiveSEModule,
    GlobalContext, GatherExcite, MHSA, S2Attention, NAMAttention, CrissCrossAttention,
    SequentialPolarizedSelfAttention, ParallelPolarizedSelfAttention, ParNetAttention,

    C2f_CBAM,C2f_CA, C2f_SE, C2f_ECA, C2f_SimAM, C2f_CoT, C2f_Double, C2f_SK,
    C2f_EffectiveSE, C2f_GlobalContext, C2f_GatherExcite, C2f_MHSA,
    C2f_Triplet, C2f_SpatialGroupEnhance, C2f_S2, C2f_NAM,
    C2f_ParNet, C2f_GAM, C2f_CrissCross, C2f_ParallelPolarized, C2f_SequentialPolarized,

    space_to_depth,

    PSA, SCDown, C2fCIB,RepVGGDW,CIB,Attention
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",

    "BasicStage", "PatchEmbed_FasterNet", "PatchMerging_FasterNet",
    'SimAM', 'ECA', 'SpatialGroupEnhance', 'TripletAttention', 'CoordAtt', 'GAMAttention',
    'SE', 'ShuffleAttention', 'SKAttention', 'DoubleAttention', 'CoTAttention', 'EffectiveSEModule',
    'GlobalContext', 'GatherExcite', 'MHSA', 'S2Attention', 'NAMAttention', 'CrissCrossAttention',
    'SequentialPolarizedSelfAttention', 'ParallelPolarizedSelfAttention', 'ParNetAttention',

    'C2f_CA', 'C2f_SE', 'C2f_ECA','C2f_CBAM','C2f_SimAM', 'C2f_CoT', 'C2f_Double', 'C2f_SK','C2f_SK',
    'C2f_EffectiveSE', 'C2f_GlobalContext', 'C2f_GatherExcite','C2f_MHSA',
    'C2f_Triplet','C2f_SpatialGroupEnhance', 'C2f_S2','C2f_NAM',
    'C2f_ParNet', 'C2f_GAM', 'C2f_CrissCross', 'C2f_ParallelPolarized', 'C2f_SequentialPolarized',
    'space_to_depth',

    'RepVGGDW',
    'CIB',
    'C2fCIB',
    'Attention',
    'PSA',
    'SCDown',
)
