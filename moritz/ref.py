# ContextNet.py (Moritz Bergemann's implementation)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Input, Add, UpSampling2D, Dropout
from tensorflow.keras import activations

# NOTES:
#   - 2 sets of brackets (second with model shortcut) are compressed constructor and method call - first brackets 
#       construct the layer object, second brackets add it to the model via its '__call__' method


def conv_block(x, filters, kernel=(3,3), stride=(1,1), do_relu=False): #FIXME does this need relu or anything?
    """
    Function representing a basic convolution with batch normalisation and ReLU6 filter.
    \nParameters:
    \nx - Layers to add to
    \nfilters - Number of channels of the output space (i.e. how many filters convolution should apply)
    \nkernel - tuple of 2, kernel size to use
    \nstride - tuple of 2, stride to use (default (1,1))
    """
    
    #Single basic convolution
    x = Conv2D(filters=filters, kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)

    if do_relu:
        x = ReLU()(x)

    return x


def depthwise_separable_conv_block(x, filters, kernel=(3, 3), stride=(1,1)):
    """
    Function representing a depthwise separable convolution. 
    \nNOTE: ReLU operation omitted between depth-wise and pointwise convolution in line with paper (see page 4)
    \nSources: https://arxiv.org/abs/1610.02357 
    \nParameters:
    \nx - Layers to add to
    \nfilters - Number of channels of the output space (i.e. how many filters should be applied during the pointwise convolution)
    \nkernel - Kernel to use during depthwise convolution (default (3,3))
    \nstride - Stride to use during depthwise convolution (default (1,1))
    """
    
    # Depthwise convolution (one for each channel)
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    # No ReLU due to little observed effect

    # Pointwise convolution (1x1) for actual new features
    x = Conv2D(filters, kernel_size=(1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x


def bottleneck_res_block_part(x, filters, expansion_factor, kernel, stride):
    """
    Function representing a bottleneck residual block. 
    \nSources: https://arxiv.org/abs/1801.04381, https://machinethink.net/blog/mobilenet-v2/
    \nParameters:
    \nx - Layers to add to
    \nfilters - Number of channels of the output space (i.e. how many filters should be applied to during the projection layer)
    \nexpansion_factor - Amount channels are expanded by in the initial expansion convolution
    \nkernel - Kernel to use for depthwise separable convolution
    \nstride - Stride to use for depthwise separable convolution
    """
    # Keeping shortcut to start for eventual joining
    x_shortcut = x
    
    ## Getting number of channels in input x (source: https://github.com/xiaochus/MobileNetV2)
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    # Depth
    initial_channels = keras.backend.int_shape(x)[channel_axis]
    expanded_channels = initial_channels * expansion_factor
    # # Width
    # cchannel = int(filters * alpha) # FIXME I don't think I need this?

    ## EXPANSION LAYER - 1x1 pointwise convolution layer (expansion factor goes here)
    x = Conv2D(filters=expanded_channels, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    ## DEPTHWISE LAYER - 3x3 depthwise convolution layer (this does the convolution work)
    x = DepthwiseConv2D(kernel_size=kernel, strides=stride, padding='same')(x) # FIXME this padding seems correct (needed for addition) but I haven't seen any explicit references to it
    x = BatchNormalization()(x) # GH implementation has this: x = BatchNormalization(axis=channel_axis)(x)
    x = ReLU()(x)

    # PROJECTION LAYER - 1x1 pointwise convolution layer (this needs to shrink it back down)
    x = Conv2D(filters=filters, kernel_size=(1, 1))(x)
    x = BatchNormalization()(x)

    # RESIDUAL CONNECTION
    # Create the residual connection if the number of input/output channels (and input/output shape) is the same
    if initial_channels == filters and stride == (1,1): #FIXME GET THIS CHECKED
        x = keras.layers.Add()([x_shortcut, x])

    return x


def bottleneck_res_block(x, filters, expansion_factor, kernel=(3, 3), stride=(1, 1), repeats=1): # sources: https://arxiv.org/abs/1801.04381 https://machinethink.net/blog/mobilenet-v2/#:~:text=The%20default%20expansion%20factor%20is,to%20that%20144%20%2Dchannel%20tensor.
    """
    Function representing a bottleneck residual block. 
    \nSources: https://arxiv.org/abs/1801.04381, https://machinethink.net/blog/mobilenet-v2/
    \nParameters:
    \nx - Layers to add to
    \nfilters - Number of channels of the output space (i.e. how many filters should be applied to during the projection layer)
    \nexpansion_factor - Amount channels are expanded by in the initial expansion convolution
    \nkernel - Kernel to use for depthwise separable convolution
    \nstride - Stride to use for depthwise separable convolution (if other than (1,1), this is only used for the first block - subsequent blocks will be (1,1))
    """
    x = bottleneck_res_block_part(x, filters, expansion_factor, kernel, stride)

    # Performing repetitions (with stride (1,1), guaranteeing residual connection)
    for i in range(1, repeats):
        x = bottleneck_res_block(x, filters, expansion_factor, kernel, (1,1))

    return x


def model(num_classes=19, input_size=(1024, 2048, 3), shrink_factor=4):
    """
    Returns an instance of the ContextNet model.
    \nParameters:
    \nnum_classes - Integer, number of classes within classification for model. (Default 19)
    \ninput_size - Tuple of 3, dimensionality of model input into shallow branch. (Default (512, 1024, 3))
    """
    ## DEEP BRANCH
    # Reference: Table on page 5 of paper
    reduced_input_size = (int(input_size[0]/shrink_factor), int(input_size[1]/shrink_factor), input_size[2]) # Reducing input size in deep branch

    deep_input = Input(shape=reduced_input_size, name='input_deep')
    deep_branch = conv_block(deep_input, filters=32, kernel=(3,3), stride=(2,2), do_relu=True)
    deep_branch = bottleneck_res_block(deep_branch, filters=32, expansion_factor=1, repeats=1)
    deep_branch = bottleneck_res_block(deep_branch, filters=32, expansion_factor=6, repeats=1)
    deep_branch = bottleneck_res_block(deep_branch, filters=48, expansion_factor=6, repeats=3, stride=(2,2))
    deep_branch = bottleneck_res_block(deep_branch, filters=64, expansion_factor=6, repeats=3, stride=(2,2))
    deep_branch = bottleneck_res_block(deep_branch, filters=96, expansion_factor=6, repeats=2)
    deep_branch = bottleneck_res_block(deep_branch, filters=128, expansion_factor=6, repeats=2)
    deep_branch = conv_block(deep_branch, filters=128, kernel=(3,3), do_relu=True) #FIXME is the kernel right?

    ## SHALLOW BRANCH
    # Reference: Page 5 - "[For the full-resolution branch] the number of feature maps are 32, 64, 128 and 128 
    #   respectively. The first layer uses standard convolution while all other layers use depth-wise separable 
    #   convolutions with kernel size 3 × 3. The stride is 2 for all but the last layer, where it is 1."
    shallow_input = Input(shape=input_size, name='input_shallow')
    shallow_branch = conv_block(shallow_input, filters=32, kernel=(3,3), stride=(2,2))
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=64, kernel=(3,3), stride=(2, 2))
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=128, kernel=(3,3), stride=(2, 2))
    shallow_branch = depthwise_separable_conv_block(shallow_branch, filters=128, kernel=(3,3), stride=(1, 1))
    
    ## FEATURE FUSION 
    # Deep branch prep
    deep_branch = UpSampling2D((4, 4))(deep_branch) # FIXME Is this correct? Paper mentions "Upsample x4"
    deep_branch = DepthwiseConv2D(kernel_size=(3,3), strides=(1,1), dilation_rate=(4,4), padding='same')(deep_branch)
    deep_branch = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same')(deep_branch)
    # Shallow branch prep
    shallow_branch = Conv2D(128, kernel_size=(1,1), strides=(1,1), padding='same')(shallow_branch)
    # Actual addition
    output = Add()([shallow_branch, deep_branch])
    
    # Dropout layer before final softmax
    # source: "During training, batch normalization is used at all layers and dropout is used before the soft-max layer only." (p6)
    output = Dropout(rate=0.35)(output)

    # Final result using number of classes
    # Source: "Finally, we use a simple 1 × 1 convolution layer for the final soft-max based classification results" (p5)
    output = Conv2D(filters=num_classes, kernel_size=(1,1), strides=(1,1), activation='softmax', name='conv_output')(output)

    # Perform upsample to return to original resolution NOTE: Not in original paper
    output = UpSampling2D((8, 8))(output)

    ## MAKING MODEL
    contextnet = Model(inputs=[shallow_input, deep_input], outputs=output)

    return contextnet


###########
## TESTS ##
###########
def print_bottleneck_res_block(input_size):
    input_stuff = Input(shape=input_size, name='bottleneck_residual_block')
    output_stuff = bottleneck_res_block(input_stuff, 3, expansion_factor=6)
    print()
    print("SHOWING BOTTLENECK RESIDUAL BLOCK:")
    model = Model(inputs=input_stuff, outputs=output_stuff)
    print(model.summary())
    keras.utils.plot_model(model, to_file="bottleneck_residual_block.png")

def print_depthwise_separable_conv_block(input_size):
    input_stuff = Input(shape=input_size, name='bottleneck_residual_block')
    output_stuff = depthwise_separable_conv_block(input_stuff, 1, kernel=(3, 3))
    print()
    print("SHOWING DEPTHWISE SEPARALE CONV BLOCK:")
    model = Model(inputs=input_stuff, outputs=output_stuff)
    print(model.summary())
    keras.utils.plot_model(model, to_file="depthwise_separable_conv_block.png")

def print_model(filename="contextnet.png"):
    contextnet = model()
    print(contextnet.summary())
    keras.utils.plot_model(contextnet, to_file=filename, show_shapes=True)

if __name__ == "__main__":
    # print_bottleneck_res_block((256, 256, 3))
    # print_depthwise_separable_conv_block((256, 256, 3))
    print_model()