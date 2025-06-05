import copy
import logging
import torch
from torch import nn
from convs.linears import SimpleLinear, SplitCosineLinear, CosineLinear
import timm
import torch.nn.functional as F
from convs.projections import Proj_Pure_MLP, MultiHeadAttention
# from models.state_evolution import InsectLifecycleModel  # 移除GCN相关导入
from utils.toolkit import get_attribute

def get_convnet(args, pretrained=False):
    backbone_name = args["convnet_type"].lower()
    algorithm_name = args["model_name"].lower()
    if 'clip' in backbone_name:
        print('Using CLIP model as the backbone')
        import open_clip
        if backbone_name == 'clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion400m_e32')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='clip_laion2b':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        elif backbone_name=='openai_clip':
            model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
            tokenizer = open_clip.get_tokenizer('ViT-B-16')
            model.out_dim = 512
            return model, preprocess, tokenizer
        else:
            raise NotImplementedError("Unknown type {}".format(backbone_name))
    else:
        raise NotImplementedError("Unknown type {}".format(backbone_name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        self.convnet = get_convnet(args, pretrained)
        self.fc = None
        self.device = args["device"][0]
        self.to(self.device)

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features,
            'logits': logits
        }
        """
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self


class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def weight_align(self, increment):
        weights = self.fc.weight.data
        newnorm = torch.norm(weights[-increment:, :], p=2, dim=1)
        oldnorm = torch.norm(weights[:-increment, :], p=2, dim=1)
        meannew = torch.mean(newnorm)
        meanold = torch.mean(oldnorm)
        gamma = meanold / meannew
        print("alignweights, gamma=", gamma)
        self.fc.weight.data[-increment:, :] *= gamma

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations
        return out

    def unset_gradcam_hook(self):
        self._gradcam_hooks[0].remove()
        self._gradcam_hooks[1].remove()
        self._gradcam_hooks[0] = None
        self._gradcam_hooks[1] = None
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

    def set_gradcam_hook(self):
        self._gradcam_gradients, self._gradcam_activations = [None], [None]

        def backward_hook(module, grad_input, grad_output):
            self._gradcam_gradients[0] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self._gradcam_activations[0] = output
            return None

        self._gradcam_hooks[0] = self.convnet.last_conv.register_backward_hook(backward_hook)
        self._gradcam_hooks[1] = self.convnet.last_conv.register_forward_hook(forward_hook)


class CosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained, nb_proxy=1):
        super().__init__(args, pretrained)
        self.nb_proxy = nb_proxy

    def update_fc(self, nb_classes, task_num):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            if task_num == 1:
                fc.fc1.weight.data = self.fc.weight.data
                fc.sigma.data = self.fc.sigma.data
            else:
                prev_out_features1 = self.fc.fc1.out_features
                fc.fc1.weight.data[:prev_out_features1] = self.fc.fc1.weight.data
                fc.fc1.weight.data[prev_out_features1:] = self.fc.fc2.weight.data
                fc.sigma.data = self.fc.sigma.data
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        if self.fc is None:
            fc = CosineLinear(in_dim, out_dim, self.nb_proxy, to_reduce=True)
        else:
            prev_out_features = self.fc.out_features // self.nb_proxy
            fc = SplitCosineLinear(in_dim, prev_out_features, out_dim - prev_out_features, self.nb_proxy)
        return fc


class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True))

    def forward(self, x, low_range, high_range):
        ret_x = x.clone()
        ret_x[:, low_range:high_range] = self.alpha * x[:, low_range:high_range] + self.beta
        return ret_x

    def get_params(self):
        return (self.alpha.item(), self.beta.item())


class IncrementalNetWithBias(BaseNet):
    def __init__(self, args, pretrained, bias_correction=False):
        super().__init__(args, pretrained)
        self.bias_correction = bias_correction
        self.bias_layers = nn.ModuleList([])
        self.task_sizes = []

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        if self.bias_correction:
            logits = out["logits"]
            for i, layer in enumerate(self.bias_layers):
                logits = layer(logits, sum(self.task_sizes[:i]), sum(self.task_sizes[:i + 1]))
            out["logits"] = logits
        out.update(x)
        return out

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc
        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)
        self.bias_layers.append(BiasLayer())

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def get_bias_params(self):
        params = []
        for layer in self.bias_layers:
            params.append(layer.get_params())
        return params

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class SimpleCosineIncrementalNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc


class SimpleVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)
        
    def forward(self, x):
        x = self.convnet.encode_image(x)
        out = self.fc(x)
        return out


class SimpleClipNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.convnet, self.preprocess, self.tokenizer = get_convnet(args, pretrained)
        self.class_name = 'SimpleClipNet'
        self.args = args

    def update_fc(self, nb_classes, nextperiod_initialization=None):
        fc = self.generate_fc(self.feature_dim, nb_classes).cuda()
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            fc.sigma.data = self.fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            fc.weight = nn.Parameter(weight)
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = CosineLinear(in_dim, out_dim)
        return fc

    def extract_vector(self, x):
        return self.convnet.encode_image(x)

    def encode_image(self, x):
        return self.convnet.encode_image(x)
    
    def encode_text(self, x):
        return self.convnet.encode_text(x)

    def forward(self, img, text):
        image_features, text_features, logit_scale = self.convnet(img, text)
        return image_features, text_features, logit_scale

    def re_initiate(self):
        print('re-initiate model')
        self.convnet, self.preprocess, self.tokenizer = get_convnet(self.args, True)


class Proof_Net(SimpleClipNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.projs_img = nn.ModuleList()
        self.projs_text = nn.ModuleList()
        self.projs_state = nn.ModuleList()  # 添加虫态投影列表
        self.args = args
        self._device = args["device"][0]
        self.projtype = get_attribute(self.args, 'projection_type', 'mlp')
        self.context_prompt_length_per_task = get_attribute(self.args, 'context_prompt_length_per_task', 3)
        
        self.sel_attn = MultiHeadAttention(1, self.feature_dim, self.feature_dim, self.feature_dim, dropout=0.1)
        self.img_prototypes = None
        self.context_prompts = nn.ParameterList()
        self.num_states = 2  
        self.state_embedding = nn.Embedding(self.num_states, self.feature_dim)
        
        # 初始化原型存储结构
        self.img_prototypes_by_state = {}

    def update_prototype(self, nb_classes):
        # 将一维原型结构改为二维结构
        if not hasattr(self, "img_prototypes_by_state"):
            self.img_prototypes_by_state = {}
        
        for class_id in range(nb_classes):
            if class_id not in self.img_prototypes_by_state:
                self.img_prototypes_by_state[class_id] = {}
        
        if self.img_prototypes is not None:
            nb_output = len(self.img_prototypes)
            self.img_prototypes = torch.cat([
                copy.deepcopy(self.img_prototypes).to(self._device),
                torch.zeros(nb_classes - nb_output, self.feature_dim).to(self._device)
            ]).to(self._device)
        else:
            self.img_prototypes = torch.zeros(nb_classes, self.feature_dim).to(self._device)
        
        print(f'更新原型，现有 {nb_classes} 个类别原型和虫态原型字典')
    
    def update_context_prompt(self):
        for i in range(len(self.context_prompts)):
            self.context_prompts[i].requires_grad = False
        self.context_prompts.append(nn.Parameter(torch.randn(self.context_prompt_length_per_task, self.feature_dim).to(self._device)))
        print('update context prompt, now we have {} context prompts'.format(len(self.context_prompts) * self.context_prompt_length_per_task))
        self.context_prompts.to(self._device)
    
    def get_context_prompts(self):
        return torch.cat([item for item in self.context_prompts], dim=0)

    def encode_image(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_img_features = self.convnet.encode_image(x)
        if not self.projs_img:
            logging.warning("encode_image called but no image projections (projs_img) are defined.")
            return F.normalize(basic_img_features, dim=-1) if normalize else basic_img_features

        img_features = [proj(basic_img_features) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)  # [bs, num_proj, dim]
        image_feas = torch.sum(img_features, dim=1)  # [bs, dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
        
    def encode_text(self, x, normalize: bool = False):
        x = x.to(self._device)
        basic_text_features = self.convnet.encode_text(x)
        if not self.projs_text: # Check if projs_text is empty
            logging.warning("encode_text called but no text projections (projs_text) are defined.")
            return F.normalize(basic_text_features, dim=-1) if normalize else basic_text_features

        text_features = [proj(basic_text_features) for proj in self.projs_text]
        text_features = torch.stack(text_features, dim=1)
        text_feas = torch.sum(text_features, dim=1)  # [bs, dim]
        return F.normalize(text_feas, dim=-1) if normalize else text_feas
        
    def encode_prototypes(self, normalize: bool = False):
        """编码原型特征，修正拼写错误"""
        self.img_prototypes = self.img_prototypes.to(self._device)
        if not self.projs_img: # Check if projs_img is empty for prototypes as well
            logging.warning("encode_prototypes called but no image projections (projs_img) are defined.")
            # Return raw prototypes or handle error
            return F.normalize(self.img_prototypes, dim=-1) if normalize else self.img_prototypes

        img_features = [proj(self.img_prototypes) for proj in self.projs_img]
        img_features = torch.stack(img_features, dim=1)  # [nb_class, num_proj, dim]
        image_feas = torch.sum(img_features, dim=1)  # [nb_class, dim]
        return F.normalize(image_feas, dim=-1) if normalize else image_feas
    
    def encode_stage_prototypes(self, class_id, stage_id, normalize: bool = False):
        """编码特定类别和阶段的原型特征 - Stage CIL核心功能"""
        if (class_id not in self.img_prototypes_by_state or 
            stage_id not in self.img_prototypes_by_state[class_id]):
            logging.warning(f"No prototype found for class {class_id}, stage {stage_id}")
            return None
            
        stage_proto = self.img_prototypes_by_state[class_id][stage_id].to(self._device)
        
        if not self.projs_img:
            logging.warning("encode_stage_prototypes called but no image projections defined.")
            return F.normalize(stage_proto, dim=-1) if normalize else stage_proto
            
        # 对单个阶段原型应用投影
        stage_features = [proj(stage_proto.unsqueeze(0)) for proj in self.projs_img]
        stage_features = torch.stack(stage_features, dim=1)  # [1, num_proj, dim]
        stage_feas = torch.sum(stage_features, dim=1).squeeze(0)  # [dim]
        return F.normalize(stage_feas, dim=-1) if normalize else stage_feas

    def extend_task(self):
        self.projs_img.append(self.extend_item())
        self.projs_text.append(self.extend_item())
        self.projs_state.append(self.extend_item())  # 为虫态添加新投影
        print(f"任务扩展: 添加新投影，当前共有 {len(self.projs_img)} 组三路投影")

    def extend_item(self):
        if self.projtype == 'pure_mlp':
            return Proj_Pure_MLP(self.feature_dim, self.feature_dim, self.feature_dim).to(self._device)
        else:
            raise NotImplementedError
    
    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)  # [bs, dim]
        text_features = self.encode_text(text, normalize=True)  # [bs, dim]
        prototype_features = self.encode_prototypes(normalize=True)  # [nb_class, dim] - 修正方法名
        context_prompts = self.get_context_prompts()  # [num_prompt, dim]

        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)  # [bs, 1, dim]
        text_features = text_features.view(text_features.shape[0], self.feature_dim)  # [num_text, dim]
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)  # [len_proto, dim]
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)  # [len_context, dim]
        text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)  # [bs, num_text, dim]
        prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)  # [bs, len_proto, dim]
        context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)  # [bs, len_context, dim]
        features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)  # [bs, (1+num_text+num_proto+num_context), dim]
        features = self.sel_attn(features, features, features)
        image_features = features[:, 0, :]  # [bs, dim]
        text_features = features[:, 1:len_texts+1, :]  # [bs, num_text, dim]
        prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :]  # [bs, num_proto, dim]
        context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :]  # [bs, num_context, dim]
        text_features = torch.mean(text_features, dim=0)  # [num_text, dim]
        prototype_features = torch.mean(prototype_features, dim=0)  # [num_proto, dim]
        image_features = image_features.view(image_features.shape[0], -1)
        text_features = text_features.view(text_features.shape[0], -1)
        prototype_features = prototype_features.view(prototype_features.shape[0], -1)
        return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def forward_transformer(self, image_features, text_features, transformer=False):
        prototype_features = self.encode_prototypes(normalize=True)  # 修正方法名
        if transformer:
            context_prompts = self.get_context_prompts()
            len_texts = text_features.shape[0]
            len_protos = prototype_features.shape[0]
            len_context_prompts = context_prompts.shape[0]
            image_features = image_features.view(image_features.shape[0], -1, self.feature_dim)  # [bs, 1, dim]
            text_features = text_features.view(text_features.shape[0], self.feature_dim)  # [total_classes, dim]
            prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)  # [len_pro, dim]
            context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)  # [len_context, dim]
            text_features = text_features.expand(image_features.shape[0], text_features.shape[0], self.feature_dim)  # [bs, total_classes, dim]
            prototype_features = prototype_features.expand(image_features.shape[0], prototype_features.shape[0], self.feature_dim)  # [bs, len_pro, dim]
            context_prompts = context_prompts.expand(image_features.shape[0], context_prompts.shape[0], self.feature_dim)  # [bs, len_context, dim]
            features = torch.cat([image_features, text_features, prototype_features, context_prompts], dim=1)
            features = self.sel_attn(features, features, features)
            image_features = features[:, 0, :]  # [bs, dim]
            text_features = features[:, 1:len_texts+1, :]  # [bs, num_text, dim]
            prototype_features = features[:, len_texts+1:len_texts+1+len_protos, :]  # [bs, num_proto, dim]
            context_prompts = features[:, len_texts+1+len_protos:len_texts+1+len_protos+len_context_prompts, :]  # [bs, num_context, dim]
            text_features = torch.mean(text_features, dim=0)  # [num_text, dim]
            prototype_features = torch.mean(prototype_features, dim=0)  # [num_proto, dim]
            image_features = image_features.view(image_features.shape[0], -1)
            text_features = text_features.view(text_features.shape[0], -1)
            prototype_features = prototype_features.view(prototype_features.shape[0], -1)
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
        else:
            return image_features, text_features, self.convnet.logit_scale.exp(), prototype_features
    
    def freeze_projection_weight_new(self):
        if len(self.projs_img) > 1:
            for i in range(len(self.projs_img) - 1):  
                for param in self.projs_img[i].parameters():
                    param.requires_grad = False
                for param in self.projs_text[i].parameters():
                    param.requires_grad = False
                for param in self.projs_state[i].parameters():  # 冻结旧的投影
                    param.requires_grad = False
            
            for param in self.projs_img[-1].parameters():
                param.requires_grad = True
            for param in self.projs_text[-1].parameters():
                param.requires_grad = True
            for param in self.projs_state[-1].parameters():
                param.requires_grad = True
        
        # 注意力模块始终可训练
        for param in self.sel_attn.parameters():
            param.requires_grad = True
        
        if self.projs_state:
            for param in self.projs_state[-1].parameters():
                param.requires_grad = True

    def encode_state(self, state_ids, normalize: bool = False):
        state_ids = torch.clamp(state_ids, 0, self.num_states - 1)
        state_features = self.state_embedding(state_ids)  # [batch_size, feature_dim]
        
        if not self.projs_state: # Check if projs_state is empty
            logging.warning("encode_state called but no state projections (projs_state) are defined.")
            # Return basic state features, possibly normalized
            return F.normalize(state_features, dim=-1) if normalize else state_features

        state_projections = [proj(state_features) for proj in self.projs_state] # Each proj output: [batch_size, feature_dim]
        # state_projections will be a list of tensors, e.g., [[bs, dim], [bs, dim], ...]
        state_features = torch.stack(state_projections, dim=1) # Should be [bs, num_proj, dim]
        state_features = torch.sum(state_features, dim=1) # Sum over num_proj => [bs, dim]
        if normalize:
            state_features = F.normalize(state_features, dim=1)
        return state_features

    def forward_tri_modal(self, image, text, state_ids):
        image_features = self.encode_image(image, normalize=True)
        if isinstance(text, list):
            text_tensor = self.tokenizer(text).to(self._device)
        else:
            text_tensor = text
        text_features = self.encode_text(text_tensor, normalize=True)
        state_features = self.encode_state(state_ids, normalize=True)
        prototype_features = self.encode_prototypes(normalize=True)  
        context_prompts = self.get_context_prompts()
        len_texts = text_features.shape[0]
        len_protos = prototype_features.shape[0]
        len_context_prompts = context_prompts.shape[0]
        batch_size = image_features.shape[0]
        image_features = image_features.view(batch_size, 1, self.feature_dim)
        state_features = state_features.view(batch_size, 1, self.feature_dim)
        if text_features.shape[0] == batch_size:
            text_features = text_features.unsqueeze(1)
        else:
            text_features = text_features.view(text_features.shape[0], self.feature_dim)
            text_features = text_features.expand(batch_size, text_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.view(prototype_features.shape[0], self.feature_dim)
        prototype_features = prototype_features.expand(batch_size, prototype_features.shape[0], self.feature_dim)
        context_prompts = context_prompts.view(context_prompts.shape[0], self.feature_dim)
        context_prompts = context_prompts.expand(batch_size, context_prompts.shape[0], self.feature_dim)
        features = torch.cat([
            image_features,
            text_features,
            state_features,
            prototype_features,
            context_prompts
        ], dim=1)
        features = self.sel_attn(features, features, features)
        image_output_idx = 0
        text_output_start_idx = 1
        if text_features.shape[1] == 1:
            text_output_end_idx = text_output_start_idx + 1
            state_output_idx = text_output_end_idx
        else:
            text_output_end_idx = text_output_start_idx + len_texts
            state_output_idx = text_output_end_idx
        proto_output_start_idx = state_output_idx + 1
        proto_output_end_idx = proto_output_start_idx + len_protos
        image_features = features[:, image_output_idx]
        text_features = features[:, text_output_start_idx:text_output_end_idx]
        state_features = features[:, state_output_idx]
        prototype_features = features[:, proto_output_start_idx:proto_output_end_idx]
        if text_features.shape[1] > 1:
            text_features = torch.mean(text_features, dim=1)
        if prototype_features.shape[1] > 1:
            prototype_features = torch.mean(prototype_features, dim=1)
        return image_features, text_features, state_features, prototype_features, self.convnet.logit_scale.exp()

    def get_all_stage_prototypes_for_class(self, class_id, normalize: bool = False):
        if class_id not in self.img_prototypes_by_state:
            return {}
            
        stage_prototypes = {}
        for stage_id, proto in self.img_prototypes_by_state[class_id].items():
            if proto is not None:
                encoded_proto = self.encode_stage_prototypes(class_id, stage_id, normalize)
                if encoded_proto is not None:
                    stage_prototypes[stage_id] = encoded_proto
        return stage_prototypes
    
    def update_stage_prototype(self, class_id, stage_id, prototype_tensor):
        if class_id not in self.img_prototypes_by_state:
            self.img_prototypes_by_state[class_id] = {}
        
        self.img_prototypes_by_state[class_id][stage_id] = prototype_tensor.clone().to(self._device)
        print(f"更新 Class {class_id}, Stage {stage_id} 原型")
    
    def get_stage_evolution_pairs(self, stage_map={0: 1, 1: None}):
        evolution_pairs = []
        for class_id, stages_dict in self.img_prototypes_by_state.items():
            for current_stage, next_stage in stage_map.items():
                if (current_stage in stages_dict and 
                    next_stage is not None and 
                    next_stage in stages_dict):
                    current_proto = stages_dict[current_stage]
                    next_proto = stages_dict[next_stage]
                    if current_proto is not None and next_proto is not None:
                        evolution_pairs.append({
                            'class_id': class_id,
                            'current_stage': current_stage,
                            'next_stage': next_stage,
                            'current_proto': current_proto,
                            'next_proto': next_proto
                        })
        return evolution_pairs
    
    def forward_stage_cil(self, image, text, state_ids, class_ids=None):
        batch_size = image.shape[0]
        
        # 基础特征提取
        image_features = self.encode_image(image, normalize=True)
        if isinstance(text, list):
            text_tensor = self.tokenizer(text).to(self._device)
        else:
            text_tensor = text
        text_features = self.encode_text(text_tensor, normalize=True)
        state_features = self.encode_state(state_ids, normalize=True)
        
        # 为每个样本获取其类别的所有阶段原型
        enhanced_features = []
        for i in range(batch_size):
            sample_image_feat = image_features[i]
            sample_text_feat = text_features[i] if text_features.shape[0] == batch_size else text_features
            sample_state_feat = state_features[i]
            
            # 如果提供了类别信息，使用该类别的阶段原型进行增强
            if class_ids is not None:
                class_id = class_ids[i].item() if torch.is_tensor(class_ids[i]) else class_ids[i]
                class_stage_protos = self.get_all_stage_prototypes_for_class(class_id, normalize=True)
                
                if class_stage_protos:
                    # 将阶段原型作为上下文
                    proto_tensors = torch.stack(list(class_stage_protos.values()), dim=0)  # [num_stages, dim]
                    proto_mean = proto_tensors.mean(dim=0)  # [dim]
                    
                    # 简单的注意力加权组合
                    stage_attention = F.softmax(torch.dot(sample_state_feat, proto_mean), dim=0)
                    enhanced_state_feat = sample_state_feat + 0.1 * stage_attention * proto_mean
                else:
                    enhanced_state_feat = sample_state_feat
            else:
                enhanced_state_feat = sample_state_feat
            
            # 三模态特征融合
            tri_modal_feat = torch.cat([
                sample_image_feat, 
                sample_text_feat if sample_text_feat.dim() == 1 else sample_text_feat.view(-1),
                enhanced_state_feat
            ], dim=0)
            enhanced_features.append(tri_modal_feat)
        
        enhanced_features = torch.stack(enhanced_features, dim=0)  # [batch_size, 3*dim]
        
        return {
            'image_features': image_features,
            'text_features': text_features,
            'state_features': state_features,
            'enhanced_features': enhanced_features,
            'logit_scale': self.convnet.logit_scale.exp()
        }


