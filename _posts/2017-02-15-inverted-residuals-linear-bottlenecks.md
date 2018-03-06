---
layout: post
title: MobileNet V2 A brief explanation and an implementation in PyTorch
---

Inverted Residuals and Linear Bottlenecks by Sandler et. al introduces us to Google's new Mobile CNN Architecture MobileNet V2. In this post I'll first talk about the current state of Mobile CNN Architectures. I'll then provide a basic explanation of grouped convolutions and finally we'll dive into the architecture of the network. This post is also supplemented by a PyTorch implementation of the original paper.

### Mobile Architectures
Mobile architectures for CNN's are designed to have the following properties:
* Small model size
* Fewer FLOPS
* Fast at testing/ inference time.

The move towards smaller architectures was started by SqueezeNet which demonstrated AlexNet level performances on ImageNet with a model size of less than 0.5MB. The next significant leap in mobile architectures was MobileNet Howard et al. This network capitalized on depthwise seperable convolutions to reduce the model size significantly whilst maintaining accuracy. ShuffleNet was another interesting architecture improving upon MobileNet. Its novelty was in reducing the parameters of pointwise convolutions by converting them into grouped pointwise convolutions and then shuffling the output channels so that the next set of grouped convolutions have groups whose inputs are nonconsecutive channels from the previous layers.

### Grouped Convolutions

![Grouped Convolutions]({{ "/assets/grouped-convolution.png" | absolute_url }})
*Figure 1: Grouped Convolutions with $$N_i$$ = C input Channels, g groups, and $$N_o$$ = C' output Channels*

To understand most recent CNN architectures, its imperative to know what grouped convolutions are. Normal convolutions apply a $$K*K$$ filter across each of the $$N_i$$ input channels to generate each output channel. This means that for a total of $$N_o$$ output channels, the kernel tensor is of shape $$(K,K,N_i,N_o)$$. As we go deeper through the network, $$K$$ usually remains the same whilst $$N_i$$ and $$N_o$$ increase rapidly. Its clear to see now, why convolutional kernels have a large number of parameters deeper in the network due to the product of $$N_i$$ and $$N_o$$ each of which are relatively large numbers. For mobile networks which look to minimize the number of parameters replacing these regular convolutions with more efficient convolutions is necessary.

Grouped convolutions try to solve this very problem. Let the number of channels in the input to the grouped convolutions be $$N_i$$, the number of desired output channels be $$N_o$$ and the group size be $$g$$ (this is an adjustable parameter). Sets of $$g$$ different input channels are selected, no channel being chosen more than once to form a group. You can think of this as slicing the input tensor along the channel dimension into $$N_i/g$$ blocks with $$g$$ channels each. For a better visual understanding of this, see Figure 1. Let's call the number of channels in each group $$N_{ig}$$ and the number of output channels for each group (divided equally) $$N_{og}$$. Now, a regular convolution is applied to each group, with each kernel of size $$(K,K,N_{ig},N_{og})$$. With $$g$$ such kernels, the total number of parameters for the entire grouped convolution is $$K*K*N_{ig}*N_{og}*g$$. Comparing this with a regular convolution with a kernel of size $$(K,K,N_i,N_o)$$, grouped convolutions have $$g$$x fewer parameters. The specific case with $$g$$ equal to $$N_i$$ and $$N_o$$ also equal to $$N_i$$ is called a depthwise convolution. In this sort of convolution, each of the $$N_i$$ filters of shape $$(K,K)$$ is applied only to one input channel and generates only one output channel.

### Bottleneck Architecture

![MobileNetV2 Architecture]({{ "/assets/mobilenet_resnet.png" | absolute_url }})
*Figure 1: MobileNetV2 Architecture, BN is used after every conv and is omitted for brevity*

Similar to all popular CNN's today including all the ones we've talked about so far, MobileNetV2 follows a residual architecture, with a unique and efficient design for each bottleneck block (Figure 1). The block starts off with an 'expansion' pointwise convolution (1x1 conv) which expands the number of channels from $$N_i$$ to $$N_i*t$$, where $$t$$ is a tuneable parameter called the expansion coefficient. A depthwise convolution of kernel size $$(K,K)$$ is applied which performs a linear transformation of the input. The last layer of this block performs a 'squeeze' pointwise convolution which squeezes the number of channels to give an output with $$N_o$$ channels. $$N_o$$ is generally either $$N_i$$ or a number far smaller than $$N_i*t$$ which is why this has been referred to as a squeezing operation.

The input is added elementwise through the residual path to the output at the last stage. In the case that the number of input channels $$N_i$$ is not equal to the number of output channels $$N_o$$ we assume that a pointwise conv is used along the residual path which has $$N_o$$ number of output channels. For bottleneck blocks with $$stride > 1$$, the stride is applied to the depthwise convolution. In this case, the width and height of the input and output map is different, hence the residual connection is omitted.

### Inverted Residuals

The original ResNet Bottleneck Block proposed by He et al. is shown in Figure 1b. At first glance, the new proposed Bottleneck block looks identical to the original one. The only difference is the number of output channels at each stage. In the original architecture, the first pointwise convolution squeezes the number of channels, and the last one expands it back to the original dimension. This is exactly the same as the proposed architecture except in an inverted order. The proposed architecture has the main benefit of lower memory consumption due to fewer channels at the residual connections.

The Bottleneck Block also has Batch Normalization after each convolutional layer and ReLU after all but the last Convolutional layer, the reason for omitting a ReLU in the last layer is explained in the next section.

### Linear Bottlenecks

The authors assume that the information present in the tensor output of any layer of the network can be encoded into a lower dimensional space. This is the motivation behind reducing the number of filters in MobileNetV1 till this lower dimensional space fully spans the output tensor itself. They then argue that using a linear activation for the bottleneck layer is necessary because a non-linearity leads to loss of information after the last pointwise convolution in the block.

In 'Indentity Mappings in Deep Residual Networks' He et. al, the proposed residual block also does not have any ReLU's either after the last convolution or after the shortcut addition. Their hypothesis was that adding a ReLU after the last convolution would lead to a non-negative tensor being propogated through the residual connection which will be monotonically increasing through the layers of the network. Indeed, both networks report a far lower testing accuracy with ReLU's after the last convolution. 

### Number of computations and paramters

| Layer     | Parameter Count |
|-----------|-----------------|
| Expansion | $$1*1*N_i*N_i*t$$   |
| Depthwise | $$3*3*N_i*t$$       |
| Squeeze   | $$1*1*N_i*t*N_o$$   |
| Total     | $$N_i*t*(N_i + 3*3 + N_o)$$   |

Similarly, the computation count is just proportional to the parameter count. It must, however, be noted that not all multiply add operations are equal in terms of inference time performance. Depthwise convolutions despite having far fewer parameters than the pointwise convolutions are the slowest operation of the block. For the next generation of mobile architectures, reducing the parameters in the 1x1 conv (like ShuffleNet) and reducing the number of depthwise conv operations to improve inference speed is required.

### Do inverted residuals work better?

An alternative design for the current block could be one where the block stared with a squeeze convolution (to reduce the number of channels for the depthwise conv) and ended with an expansion convolution much like the original ResNet architecture, however not only does this lead to a far larger network in memory but it is also detrimental to performance as shown in the figure below. In our own experiments on the CIFAR10 dataset, we see that the performance does in fact degrade as shown below. (actually run this experiment)

### Do linear bottlenecks work better?

Yes, for this architecture they do. The general idea from He et al's 'Identitiy Mappings in Deep Residual Networks' seems to apply here too, adding a non-linearity (ReLU) at the end of the residual path or at the end of the bottleneck hurts performance. Though, interestingly, a full pre-activation (BN and ReLU at the beginnining of the residual path) also proposed in the same paper which significantly boosted performance of ResNet, does not seem to help with accuracy here.

In the chart below, Baseline refers to the original MobileNetV2 architecture proposed in the paper, the Pre Activation Bottleneck starts with a BN followed by ReLU and Bottleneck ReLU as the name implies has a ReLU at the end of the bottleneck. 

![Activation Position Comparison]({{ "/assets/activation_position.png" | absolute_url }})
*Figure 3: A comparison amongst the afformentioned Baseline, Pre Activation and Bottleneck ReLU architectures*

### What is the correct value for the expansion hyperparameter?

The authors indicate that performance is more or less consistent across values for $$t$$ between 4 and 10. Our experiments on CIFAR10 concur with this assesment. For values of $$t$$ below 4, accuracy rapidly drops.

![Expansion Hyperparameter Comparison]({{ "/assets/expansion_comparison.png" | absolute_url }})
*Figure 3: A comparison between different values of the expasion hyperparameter $$t$$*

### PyTorch Implementation

A complete implementation is available on this GitHub link.

{% highlight python %}
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_coeff=1, stride=1):
        super(Bottleneck, self).__init__()
        
        t = expansion_coeff
        self.is_residual = stride == 1

        self.conv_expand = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*t, stride=1, kernel_size=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(num_features=in_channels*t)
        self.relu_expand = nn.ReLU6(inplace=True)
        

        self.conv_depthwise = nn.Conv2d(in_channels=in_channels*t, out_channels=in_channels*t, groups=in_channels*t, stride=stride, kernel_size=3, padding=1, bias=False)
        self.bn_depthwise = nn.BatchNorm2d(num_features=in_channels*t)
        self.relu_depthwise = nn.ReLU6(inplace=True)
        
        self.conv_squeeze = nn.Conv2d(in_channels=in_channels*t, out_channels=out_channels, stride=1, kernel_size=1, padding=0, bias=False)

        self.conv_res = nn.Sequential()
        if self.is_residual and in_channels != out_channels:
            self.conv_res = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))

    def forward(self, x):

        out = self.conv_expand.forward(x)
        out = self.bn_expand.forward(out)
        out = self.relu_expand.forward(out)

        out = self.conv_depthwise.forward(out)
        out = self.bn_depthwise.forward(out)
        out = self.relu_depthwise.forward(out)

        out = self.conv_squeeze.forward(out)
        
        if self.is_residual:
            x = self.conv_res.forward(x)
            out += x
        
        return out

class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=s, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.group1 = self.block('group_1', in_channels=32, out_channels=16, block_depth=1)
        self.group2 = self.block('group_2', in_channels=16, out_channels=24, block_depth=2, expansion_coeff=6, stride=1)
        self.group3 = self.block('group_3', in_channels=24, out_channels=32, block_depth=3, expansion_coeff=6, stride=2)
        self.group4 = self.block('group_4', in_channels=32, out_channels=64, block_depth=4, expansion_coeff=6, stride=2)
        self.group5 = self.block('group_5', in_channels=64, out_channels=96, block_depth=3, expansion_coeff=6, stride=1)
        self.group6 = self.block('group_6', in_channels=96, out_channels=160, block_depth=3, expansion_coeff=6, stride=2)
        self.group7 = self.block('group_7', in_channels=160, out_channels=320, block_depth=1, expansion_coeff=6, stride=1)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AvgPool2d(4)
        self.conv3 = nn.Conv2d(1280, 10, kernel_size=1, stride=1, bias=True)

    def block(self, name, in_channels, out_channels, block_depth, expansion_coeff=1, stride=1):
        block = nn.Sequential()
        for bottlenext in range(block_depth):
            name = '{}_bottleneck_{}'.format(name, bottlenext)
            s = 1
            i = out_channels
            if bottlenext == 0:
                s = stride
                i = in_channels
            block.add_module(name, Bottleneck(i, out_channels, expansion_coeff=expansion_coeff, stride=s))
        return block

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = F.relu(x, inplace=True)
        
        x = self.group1.forward(x)
        x = self.group2.forward(x)
        x = self.group3.forward(x)
        x = self.group4.forward(x)
        x = self.group5.forward(x)
        x = self.group6.forward(x)
        x = self.group7.forward(x)

        x = self.conv2.forward(x)
        x = self.bn2.forward(x)
        x = F.relu(x, inplace=True)

        x = self.avgpool.forward(x)
        x = self.conv3.forward(x)

        x = x.view(-1, 10)
        return x
{% endhighlight %}

### References

* [1] M Sandler et al. Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation

* [2] FN Iandola et al. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size

* [3] AG Howard et al. MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications

* [4] X Zhang et al. ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices

* [5] K He et al. Deep Residual Learning for Image Recognition

* [6] K He et al. Identitiy Mappings in Deep Residual Networks

