# For linter, remove on execute
# from PlotNeuralNet.pycore.tikzeng import *
# from PlotNeuralNet.pycore.blocks import *

import sys
sys.path.append('../')
from pycore.tikzeng import *
from pycore.blocks  import *

arch = [ 
    to_head( '..' ),
    to_cor(),
    to_begin(),


    # Inputs (NOT TO SCALE)
    to_input('img/frankfurt_000000_000294_leftImg8bit.png', to="(-8,-10,0)", width=16, height=8),
    to_input('img/frankfurt_000000_000294_leftImg8bit.png', to="(-5,5,0)", width=8, height=4),

    # SHALLOW BRANCH
    # Pretend the start is 256,128,3
    to_ContextnetConv(name="shallow_conv_start",  depth=128,  height=64,  width=4,    to="(0,-10,0)",                 offset="(0,0,0)",   s_filer="512x1024"),

    to_DSConv(name="dsconv_a",            depth=64,   height=32,  width=8,    to="(shallow_conv_start-east)", offset="(10,0,0)",   s_filer="256x512"),

    to_DSConv(name="dsconv_b1",           depth=32,   height=16,  width=16,   to="(dsconv_a-east)",           offset="(10,0,0)"),
    to_DSConv(name="dsconv_b2",           depth=32,   height=16,  width=16,   to="(dsconv_b1-east)",          offset="(2,0,0)",   s_filer="128x256"),

    to_connection("shallow_conv_start", "dsconv_a"),
    to_connection("dsconv_a",           "dsconv_b1"),


    # DEEP BRANCH
    # Pretend the start is 64,32,3
    to_ContextnetConv(name="deep_conv_start", depth=32,   height=16,  width=4,   to="(0,5,0)",                   offset="(0,0,0)",   s_filer="256x512"),

    to_BottleneckRes(name="bottleneck_a1",   depth=32,   height=16,  width=4,   to="(deep_conv_start-east)",    offset="(2,0,0)"),
    to_BottleneckRes(name="bottleneck_a2",   depth=32,   height=16,  width=4,   to="(bottleneck_a1-east)",      offset="(0.2,0,0)", s_filer="256x512"),

    to_BottleneckRes(name="bottleneck_b1",   depth=16,   height=8,   width=6,   to="(bottleneck_a2-east)",      offset="(2,0,0)"),
    to_BottleneckRes(name="bottleneck_b2",   depth=16,   height=8,   width=6,   to="(bottleneck_b1-east)",      offset="(0.2,0,0)"),
    to_BottleneckRes(name="bottleneck_b3",   depth=16,   height=8,   width=6,   to="(bottleneck_b2-east)",      offset="(0.2,0,0)", s_filer="128x256"),

    to_BottleneckRes(name="bottleneck_c1",   depth=8,    height=4,   width=8,   to="(bottleneck_b2-east)",      offset="(4,0,0)"),
    to_BottleneckRes(name="bottleneck_c2",   depth=8,    height=4,   width=8,   to="(bottleneck_c1-east)",      offset="(0.2,0,0)"),
    to_BottleneckRes(name="bottleneck_c3",   depth=8,    height=4,   width=8,   to="(bottleneck_c2-east)",      offset="(0.2,0,0)", s_filer="64x128"),

    to_BottleneckRes(name="bottleneck_d1",   depth=4,    height=2,   width=12,   to="(bottleneck_c3-east)",      offset="(2,0,0)"),
    to_BottleneckRes(name="bottleneck_d2",   depth=4,    height=2,   width=12,   to="(bottleneck_d1-east)",      offset="(0.2,0,0)", s_filer="32x64"),

    to_BottleneckRes(name="bottleneck_e1",   depth=4,    height=2,   width=16,  to="(bottleneck_d2-east)",      offset="(2,0,0)"),
    to_BottleneckRes(name="bottleneck_e2",   depth=4,    height=2,   width=16,  to="(bottleneck_e1-east)",      offset="(0.2,0,0)", s_filer="32x64"),

    to_ContextnetConv(name="deep_conv_end",   depth=4,    height=2,   width=16,  to="(bottleneck_e1-east)",      offset="(5,0,0)",   s_filer="32x64"),

    to_connection("deep_conv_start",    "bottleneck_a1"),
    to_connection("bottleneck_a2",      "bottleneck_b1"),
    to_connection("bottleneck_b3",      "bottleneck_c1"),
    to_connection("bottleneck_c3",      "bottleneck_d1"),
    to_connection("bottleneck_d2",      "bottleneck_e2"),
    to_connection("bottleneck_e2",      "deep_conv_end"),
    

    # FINAL CONNECTION
    to_Sum(name="concat", to="(43,0,0)"),

    to_point("deep_point2",     to="(42,5,0)"),
    to_point("shallow_point2",  to="(42,-10,0)"),

    to_ContextnetConv(name="conv_final", depth=32, height=16, width=16,  to="(concat-east)", offset="(5,0,0)", s_filer="32x64"),

    to_connection("deep_conv_end",  "deep_point2"),
    to_connection("dsconv_b2",      "shallow_point2"),
    to_connection("deep_point2", "concat"),
    to_connection("shallow_point2", "concat"),
    to_connection("concat", "conv_final"),

    to_input("img/frankfurt_000000_000294_gtFine_color.png", width=16, height=8, to="(57,0,0)"),

    to_end()
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    


