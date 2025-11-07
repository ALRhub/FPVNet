import einops
import torch
import torch.nn as nn
from timm import create_model


class FiLMLayer(nn.Module):
    def __init__(self, num_features, condition_dim):
        super(FiLMLayer, self).__init__()
        self.num_features = num_features
        self.condition_dim = condition_dim
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)

        # Zero initialization
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, condition):
        gamma = self.gamma(condition).unsqueeze(2).unsqueeze(3)
        beta = self.beta(condition).unsqueeze(2).unsqueeze(3)
        x = (1 + gamma) * x + beta  # Using (1 + gamma) to start with identity transform
        return x.contiguous()


class FiLMResNet50Policy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet50Policy, self).__init__()
        # Load pretrained ResNet50 with weights from ImageNet-1K
        self.resnet = create_model('resnet50', pretrained=True, num_classes=0)

        # Add FiLM layers after each residual block
        self.film1 = FiLMLayer(256, condition_dim)
        self.film2 = FiLMLayer(512, condition_dim)
        self.film3 = FiLMLayer(1024, condition_dim)
        self.film4 = FiLMLayer(2048, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        return x  # Return the latent features directly


class FiLMResNet34Policy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet34Policy, self).__init__()
        # Load pretrained ResNet34 with weights from ImageNet-1K
        self.resnet = create_model('resnet34', pretrained=True, num_classes=0)

        # Add FiLM layers after each residual block
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        return x  # Return the latent features directly
    
class FiLMResNet34LocalPolicy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet34LocalPolicy, self).__init__()
        # Load pretrained ResNet34 with weights from ImageNet-1K
        self.resnet = create_model('resnet34', pretrained=True, num_classes=0)

        # Add FiLM layers after each residual block
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x) # b 64 64 64
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x) # b 64 32 32

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x) # b 128 16 16
        x = self.film2(x, condition)

        x = self.resnet.layer3(x) # b 256 8 8
        x = self.film3(x, condition)

        x = self.resnet.layer4(x) # b 512 4 4
        x = self.film4(x, condition)

        local_features = einops.rearrange(x, "b c h w -> b (h w) c")
        
        x = self.resnet.global_pool(x) # b 512
        x = einops.rearrange(x, "b c -> b 1 c")

        return torch.cat([x, local_features], dim=1) 


class FiLMResNet18Policy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet18Policy, self).__init__()
        # Load pretrained ResNet18 with weights from ImageNet-1K
        self.resnet = create_model('resnet18', pretrained=True, num_classes=0)

        # Add FiLM layers after each residual block
        # ResNet18 has the same channel dimensions as ResNet34
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        x = self.resnet.global_pool(x)
        x = x.flatten(1)

        return x  # Return the latent features directly
    
class FiLMResNet18LocalPolicy(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet18LocalPolicy, self).__init__()
        # Load pretrained ResNet18 with weights from ImageNet-1K
        self.resnet = create_model('resnet18', pretrained=True, num_classes=0)

        # Add FiLM layers after each residual block
        # ResNet18 has the same channel dimensions as ResNet34
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        local_features = einops.rearrange(x, "b c h w -> b (h w) c")
        
        x = self.resnet.global_pool(x)
        x = einops.rearrange(x, "b c -> b 1 c")

        return torch.cat([x, local_features], dim=1)
    
class FiLMResNet18LocalPolicy8x8(nn.Module):
    def __init__(self, condition_dim):
        super(FiLMResNet18LocalPolicy8x8, self).__init__()
        # Load pretrained ResNet18 with weights from ImageNet-1K
        self.resnet = create_model('resnet18', pretrained=True, num_classes=0)

        # Add FiLM layers after each residual block
        # ResNet18 has the same channel dimensions as ResNet34
        self.film1 = FiLMLayer(64, condition_dim)
        self.film2 = FiLMLayer(128, condition_dim)
        self.film3 = FiLMLayer(256, condition_dim)
        self.film4 = FiLMLayer(512, condition_dim)

        # self.global_to_local_dim_model = nn.Linear(512, 256)
        self.local_to_global_dim_model = nn.Linear(256, 512)

    def forward(self, x, condition):
        if len(condition.shape) == 3:
            condition = condition.squeeze(1)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.act1(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.film1(x, condition)

        x = self.resnet.layer2(x)
        x = self.film2(x, condition)

        x = self.resnet.layer3(x)
        x = self.film3(x, condition)

        local_features = einops.rearrange(x, "b c h w -> b (h w) c")
        local_features = self.local_to_global_dim_model(local_features)

        x = self.resnet.layer4(x)
        x = self.film4(x, condition)

        x = self.resnet.global_pool(x)
        x = einops.rearrange(x, "b c -> b 1 c")

        # x = self.global_to_local_dim_model(x)

        return torch.cat([x, local_features], dim=1)