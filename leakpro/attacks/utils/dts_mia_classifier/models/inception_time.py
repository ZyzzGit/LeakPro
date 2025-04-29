import torch
import torch.nn as nn

DEFAULT_MAX_KERNEL_SIZE = 40

class GlobalAveragePooling1D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=2)

class ShortcutLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same', bias=False)
        self.bnorm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        shortcut = self.conv(x)
        shortcut = self.bnorm(shortcut)
        x = shortcut + y
        x = self.relu(x)
        return x

class InceptionModule(nn.Module):
    def __init__(self, use_bottleneck, bottleneck_size, in_channels, num_filters, max_kernel_size, fixed_kernel_sizes=None):
        super().__init__()
        if use_bottleneck and in_channels > 1:
            self.input_layer = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, stride=1, padding='same', bias=False)
        else:
            self.input_layer = nn.Identity()

        if fixed_kernel_sizes:
            self.kernel_sizes = fixed_kernel_sizes  # Use custom kernel sizes if supplied
        else:
            self.kernel_sizes = [
                k if k % 2 == 1 else k - 1  # Ensure odd kernel sizes for performance (not in original InceptionTime)
                for k in [max_kernel_size // (2 ** i) for i in range(3)]
            ]

        self.convs = nn.ModuleList()
        inception_in_channels = bottleneck_size if use_bottleneck else in_channels
        for kernel_size in self.kernel_sizes:
            self.convs.append(nn.Conv1d(inception_in_channels, num_filters, kernel_size=kernel_size, stride=1, padding='same', bias=False))

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)    # padding=1 results in 'same' padding for kernel_size=3, stride=1
        self.mp_conv = nn.Conv1d(in_channels, num_filters , kernel_size=1, stride=1, padding='same', bias=False)

        num_convs = len(fixed_kernel_sizes) + 1 if fixed_kernel_sizes else 4
        num_channels_after_concat = num_filters * num_convs
        self.bnorm = nn.BatchNorm1d(num_channels_after_concat)

        self.relu = nn.ReLU()

    def forward(self, x):
        inception_input = self.input_layer(x)
        conv_list_out = [conv(inception_input) for conv in self.convs]
        mp_conv_out = self.mp_conv(self.max_pool(x))
        conv_list_out.append(mp_conv_out)
        x = torch.cat(conv_list_out, dim=1)
        x = self.bnorm(x)
        x = self.relu(x)
        return x

class InceptionTime(nn.Module):
    def __init__(
            self, 
            in_channels, 
            num_filters=32, 
            use_residual=True, 
            use_bottleneck=True, 
            bottleneck_size=32, 
            depth=6, 
            max_kernel_size=DEFAULT_MAX_KERNEL_SIZE, 
            fixed_kernel_sizes=None
        ):
        """By default inits with original InceptionTime params, however:
            > while original InceptionModule utilizes 40, 20, respectively 10 kernel sizes, this implementation ensures only odd ones (39, 19, 9)
            > max_kernel_size (see lines 44-46) may be overriden by fixed_kernel_sizes
            > fixed_kernel_sizes lets user specify exact list of kernel sizes to use in InceptionModule (warning: check that max kernel size do not exceed sequence length)
        """
        super().__init__()
        self.use_residual = use_residual

        num_inception_module_convs = len(fixed_kernel_sizes) + 1 if fixed_kernel_sizes else 4
        num_channels_after_concat = num_filters * num_inception_module_convs

        self.inception_modules = nn.ModuleList([
            InceptionModule(use_bottleneck, bottleneck_size, in_channels, num_filters, max_kernel_size, fixed_kernel_sizes),
            *[InceptionModule(use_bottleneck, bottleneck_size, num_channels_after_concat, num_filters, max_kernel_size, fixed_kernel_sizes) for _ in range(1, depth)]
        ])

        if use_residual:
            num_shortcuts = len([d for d in range(depth) if d % 3 == 2])
            self.shortcuts = nn.ModuleList([
                ShortcutLayer(in_channels, num_channels_after_concat),
                *[ShortcutLayer(num_channels_after_concat, num_channels_after_concat) for _ in range(1, num_shortcuts)]
            ])
        
        self.gap = GlobalAveragePooling1D()
        self.fc = nn.Linear(num_channels_after_concat, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # channels first
        input_residual = x
        for d, inception_module in enumerate(self.inception_modules):
            x = inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self.shortcuts[d // 3](input_residual, x)
                input_residual = x

        x = self.gap(x)
        x = self.fc(x)
        x = self.sigmoid(x) # use sigmoid to model membership probability (original InceptionTime uses softmax to support arbitrary amount of classes)
        return x