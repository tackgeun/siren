'''Modules for hypernetwork experiments, Paper Sec. 4.4
'''

import math
import torch
import torchvision
from torch import nn
import numpy as np
from collections import OrderedDict
from typing import Any, Callable, Dict, Mapping, Optional, Tuple
import modules
import pdb

class HyperNetwork(nn.Module):
    def __init__(self, hyper_in_features, hyper_hidden_layers, hyper_hidden_features, hypo_module):
        '''

        Args:
            hyper_in_features: In features of hypernetwork
            hyper_hidden_layers: Number of hidden layers in hypernetwork
            hyper_hidden_features: Number of hidden units in hypernetwork
            hypo_module: MetaModule. The module whose parameters are predicted.
        '''
        super().__init__()

        hypo_parameters = hypo_module.meta_named_parameters()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.size())

            hn = modules.FCBlock(in_features=hyper_in_features, out_features=int(torch.prod(torch.tensor(param.size()))),
                                 num_hidden_layers=hyper_hidden_layers, hidden_features=hyper_hidden_features,
                                 outermost_linear=True, nonlinearity='relu')
            self.nets.append(hn)

            if 'weight' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_weight_init(m, param.size()[-1]))
            elif 'bias' in name:
                self.nets[-1].net[-1].apply(lambda m: hyper_bias_init(m))

    def forward(self, z):
        '''
        Args:
            z: Embedding. Input to hypernetwork. Could be output of "Autodecoder" (see above)

        Returns:
            params: OrderedDict. Can be directly passed as the "params" parameter of a MetaModule.
        '''
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)
        return params


class NeuralProcessImplicit2DHypernet(nn.Module):
    '''A canonical 2D representation hypernetwork mapping 2D coords to out_features.'''
    def __init__(self, in_features, out_features, image_resolution=None, encoder_nl='sine'):
        super().__init__()

        latent_dim = 256
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        self.set_encoder = modules.SetEncoder(in_features=in_features, out_features=latent_dim, num_hidden_layers=2,
                                              hidden_features=latent_dim, nonlinearity=encoder_nl)
        print(self)

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def get_hypo_net_weights(self, model_input):
        pixels, coords = model_input['img_sub'], model_input['coords_sub']
        ctxt_mask = model_input.get('ctxt_mask', None)
        embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            pixels, coords = model_input['img_sub'], model_input['coords_sub']
            ctxt_mask = model_input.get('ctxt_mask', None)
            embedding = self.set_encoder(coords, pixels, ctxt_mask=ctxt_mask)
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)
        return {'model_in':model_output['model_in'], 'model_out':model_output['model_out'], 'latent_vec':embedding,
                'hypo_params':hypo_params}


class ConvolutionalNeuralProcessImplicit2DHypernet(nn.Module):
    def __init__(self, in_features, out_features, image_resolution=None, partial_conv=False):
        super().__init__()
        latent_dim = 256

        if partial_conv:
            self.encoder = modules.PartialConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        else:
            self.encoder = modules.ConvImgEncoder(channel=in_features, image_resolution=image_resolution)
        self.hypo_net = modules.SingleBVPNet(out_features=out_features, type='sine', sidelength=image_resolution,
                                             in_features=2)
        self.hyper_net = HyperNetwork(hyper_in_features=latent_dim, hyper_hidden_layers=1, hyper_hidden_features=256,
                                      hypo_module=self.hypo_net)
        print(self)

    def forward(self, model_input):
        if model_input.get('embedding', None) is None:
            embedding = self.encoder(model_input['img_sparse'])
        else:
            embedding = model_input['embedding']
        hypo_params = self.hyper_net(embedding)

        model_output = self.hypo_net(model_input, params=hypo_params)

        return {'model_in': model_output['model_in'], 'model_out': model_output['model_out'], 'latent_vec': embedding,
                'hypo_params': hypo_params}

    def get_hypo_net_weights(self, model_input):
        embedding = self.encoder(model_input['img_sparse'])
        hypo_params = self.hyper_net(embedding)
        return hypo_params, embedding

    def freeze_hypernet(self):
        for param in self.hyper_net.parameters():
            param.requires_grad = False
        for param in self.encoder.parameters():
            param.requires_grad = False


###############################
# Functa baseline and variants
###############################
class FiLM(nn.Module):
  """Applies a FiLM modulation: out = scale * in + shift.

  Notes:
    We currently initialize FiLM layers as the identity. However, this may not
    be optimal. In pi-GAN for example they initialize the layer with a random
    normal.
  """

  def __init__(self,
               f_in: int,
               modulate_scale: bool = True,
               modulate_shift: bool = True):
    """Constructor.

    Args:
      f_in: Number of input features.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
    """
    super().__init__()
    # Must modulate at least one of shift and scale
    assert modulate_scale or modulate_shift
    self.f_in = f_in
    # Initialize FiLM layers as identity
    self.scale = 1.
    self.shift = 0.
    if modulate_scale:
      self.scale = nn.Parameter(torch.ones(self.f_in))
    if modulate_shift:
      self.shift = nn.Parameter(torch.zeros(self.f_in))

  def forward(self, x):
    return self.scale * x + self.shift


class ModulatedSirenLayer(nn.Module):
  """Applies a linear layer followed by a modulation and sine activation."""

  def __init__(self,
               f_in: int,
               f_out: int,
               w0: float = 1.,
               is_first: bool = False,
               is_last: bool = False,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               apply_activation: bool = True):
    """Constructor.

    Args:
      f_in (int): Number of input features.
      f_out (int): Number of output features.
      w0 (float): Scale factor in sine activation.
      is_first (bool): Whether this is first layer of model.
      is_last (bool): Whether this is last layer of model.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
      apply_activation: If True, applies sine activation.
    """
    super().__init__()
    self.f_in = f_in
    self.f_out = f_out
    self.w0 = w0
    self.is_first = is_first
    self.is_last = is_last
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift
    self.apply_activation = apply_activation
    # Follow initialization scheme from SIREN
    self.init_range = 1 / f_in if is_first else math.sqrt(6 / f_in) / w0

    self.w = torch.nn.Linear(self.f_in, self.f_out)

    if self.modulate_scale or self.modulate_shift:
      self.FiLM = FiLM(self.f_out,
                      modulate_scale=self.modulate_scale,
                      modulate_shift=self.modulate_shift)

    with torch.no_grad():
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        self.w.weight.uniform_(-self.init_range, self.init_range)


  def forward(self, x):
    # Shape (n, f_in) -> (n, f_out)
    x = self.w(x)
    # Apply non-linearities
    if self.is_last:
      # We assume target data (e.g. RGB values of pixels) lies in [0, 1]. To
      # learn zero-centered features we therefore shift output by .5
      return x + .5
    else:
      # Optionally apply modulation
      if self.modulate_scale or self.modulate_shift:
        x = self.FiLM(x)
      # Optionally apply activation
      if self.apply_activation:
        x = torch.sin(self.w0 * x)
      return x


class LatentVector(nn.Module):
  """Module that holds a latent vector.

  Notes:
    This module does not apply any transformation but simply stores a latent
    vector. This is to make sure that all data necessary to represent an image
    (or a NeRF scene or a video) is present in the model params. This also makes
    it easier to use the partition_params function.
  """

  def __init__(self, latent_dim: int, latent_init_scale: float = 0.0):
    """Constructor.

    Args:
      latent_dim: Dimension of latent vector.
      latent_init_scale: Scale at which to randomly initialize latent vector.
    """
    super().__init__()
    self.latent_dim = latent_dim
    self.latent_init_scale = latent_init_scale
    # Initialize latent vector
    self.latent_vector = nn.Parameter(torch.zeros(latent_dim).uniform_(-latent_init_scale,latent_init_scale))

  def forward(self):
    return self.latent_vector


class LatentToModulation(nn.Module):
  """Function mapping latent vector to a set of modulations."""

  def __init__(self,
               latent_dim: int,
               latent_vector_type: str,
               layer_sizes: Tuple[int, ...],
               width: int,
               num_modulation_layers: int,
               modulate_scale: bool = True,
               modulate_shift: bool = True,
               activation: str = 'relu'):
    """Constructor.

    Args:
      latent_dim: Dimension of latent vector (input of LatentToModulation
        network).
      layer_sizes: List of hidden layer sizes for MLP parameterizing the map
        from latent to modulations. Input dimension is inferred from latent_dim
        and output dimension is inferred from number of modulations.
      width: Width of each hidden layer in MLP of function rep.
      num_modulation_layers: Number of layers in MLP that contain modulations.
      modulate_scale: If True, returns scale modulations.
      modulate_shift: If True, returns shift modulations.
      activation: Activation function to use in MLP.
    """
    super().__init__()
    # Must modulate at least one of shift and scale
    assert modulate_scale or modulate_shift

    self.latent_dim = latent_dim
    self.latent_vector_type = latent_vector_type
    self.layer_sizes = tuple(layer_sizes)  # counteract XM that converts to list
    self.width = width
    self.num_modulation_layers = num_modulation_layers
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift

    # MLP outputs all modulations. We apply modulations on every hidden unit
    # (i.e on width number of units) at every modulation layer.
    # At each of these we apply either a scale or a shift or both,
    # hence total output size is given by following formula
    self.modulations_per_unit = int(modulate_scale) + int(modulate_shift)
    self.modulations_per_layer = width * self.modulations_per_unit
    self.output_size = num_modulation_layers * self.modulations_per_layer

    #self.mlp = torchvision.ops.MLP(hidden_channels=self.layer_sizes + (self.output_size,))
    layer_sizes = (self.latent_dim,) + self.layer_sizes + (self.output_size,)
    mlp = []
    for i in range(0, len(layer_sizes)-1):
      mlp.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
    self.mlp = nn.Sequential(*mlp)

  def forward(self, latent_vector: torch.Tensor) -> Dict[int, Dict[str, torch.Tensor]]:
    modulations = self.mlp(latent_vector)
    # Partition modulations into scales and shifts at every layer
    outputs = {}

    if self.latent_vector_type == 'shared':
      for i in range(self.num_modulation_layers):
        single_layer_modulations = {}
        # Note that we add 1 to scales so that outputs of MLP will be centered
        # (since scale = 1 corresponds to identity function)
        if self.modulate_scale and self.modulate_shift:
          start = 2 * self.width * i
          single_layer_modulations['scale'] = modulations[start:start +
                                                          self.width] + 1
          single_layer_modulations['shift'] = modulations[start +
                                                          self.width:start +
                                                          2 * self.width]
        elif self.modulate_scale:
          start = self.width * i
          single_layer_modulations['scale'] = modulations[start:start +
                                                          self.width] + 1
        elif self.modulate_shift:
          start = self.width * i
          single_layer_modulations['shift'] = modulations[start:start +
                                                          self.width]
        outputs[i] = single_layer_modulations

    elif self.latent_vector_type == 'instance':
      for i in range(self.num_modulation_layers):
        single_layer_modulations = {}
        # Note that we add 1 to scales so that outputs of MLP will be centered
        # (since scale = 1 corresponds to identity function)
        if self.modulate_scale and self.modulate_shift:
          start = 2 * self.width * i
          single_layer_modulations['scale'] = modulations[:, start:start +
                                                          self.width].unsqueeze(1) + 1
          single_layer_modulations['shift'] = modulations[:, start +
                                                          self.width:start +
                                                          2 * self.width].unsqueeze(1)
        elif self.modulate_scale:
          start = self.width * i
          single_layer_modulations['scale'] = modulations[:, start:start +
                                                          self.width].unsqueeze(1) + 1
        elif self.modulate_shift:
          start = self.width * i
          single_layer_modulations['shift'] = modulations[:, start:start +
                                                          self.width].unsqueeze(1)
        outputs[i] = single_layer_modulations        
    return outputs

class MetaSGDLrs(nn.Module):
  """Module storing learning rates for meta-SGD.

  Notes:
    This module does not apply any transformation but simply stores the learning
    rates. Since we also learn the learning rates we treat them the same as
    model params.
  """

  def __init__(self,
               num_lrs: int,
               lrs_init_range: Tuple[float, float] = (0.005, 0.1),
               lrs_clip_range: Tuple[float, float] = (-5., 5.)):
    """Constructor.

    Args:
      num_lrs: Number of learning rates to learn.
      lrs_init_range: Range from which initial learning rates will be
        uniformly sampled.
      lrs_clip_range: Range at which to clip learning rates. Default value will
        effectively avoid any clipping, but typically learning rates should
        be positive and small.
    """
    super().__init__()
    self.num_lrs = num_lrs
    self.lrs_init_range = lrs_init_range
    self.lrs_clip_range = lrs_clip_range
    # Initialize learning rates
    self.meta_sgd_lrs = torch.nn.Parameter(torch.zeros(self.num_lrs).uniform_(self.lrs_init_range[0], self.lrs_init_range[1]))

  def forward(self):
    # Clip learning rate values
    self.meta_sgd_lrs = torch.clamp(self.meta_sgd_lrs, self.lrs_clip_range[0], self.lrs_clip_range[1])
    return self.meta_sgd_lrs


class LatentModulatedSiren(nn.Module):
  """SIREN model with FiLM modulations generated from a latent vector."""

  def __init__(self,
               width: int = 256,
               depth: int = 5,
               out_channels: int = 3,
               latent_dim: int = 128,
               latent_vector_type: str = 'shared',
               layer_sizes: Tuple[int, ...] = (),
               w0: float = 30.,
               modulate_scale: bool = False,
               modulate_shift: bool = True,
               latent_init_scale: float = 0.01,
               use_meta_sgd: bool = False,
               meta_sgd_init_range: Tuple[float, float] = (0.005, 0.1),
               meta_sgd_clip_range: Tuple[float, float] = (-5., 5.)):
    """Constructor.

    Args:
      width (int): Width of each hidden layer in MLP.
      depth (int): Number of layers in MLP.
      out_channels (int): Number of output channels.
      latent_dim: Dimension of latent vector (input of LatentToModulation
        network).
      layer_sizes: List of hidden layer sizes for MLP parameterizing the map
        from latent to modulations. Input dimension is inferred from latent_dim
        and output dimension is inferred from number of modulations.
      w0 (float): Scale factor in sine activation in first layer.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
      latent_init_scale: Scale at which to randomly initialize latent vector.
      use_meta_sgd: Whether to use meta-SGD.
      meta_sgd_init_range: Range from which initial meta_sgd learning rates will
        be uniformly sampled.
      meta_sgd_clip_range: Range at which to clip learning rates.
    """
    super().__init__()
    self.width = width
    self.depth = depth
    self.in_channels = 2
    self.out_channels = out_channels
    self.latent_dim = latent_dim
    self.layer_sizes = layer_sizes
    self.w0 = w0
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift
    self.latent_init_scale = latent_init_scale
    self.latent_vector_type = latent_vector_type
    self.use_meta_sgd = use_meta_sgd
    self.meta_sgd_init_range = meta_sgd_init_range
    self.meta_sgd_clip_range = meta_sgd_clip_range

    # Initialize meta-SGD learning rates
    if self.use_meta_sgd:
      self.meta_sgd_lrs = MetaSGDLrs(self.latent_dim,
                                     self.meta_sgd_init_range,
                                     self.meta_sgd_clip_range)

    # Initialize latent vector and map from latents to modulations
    self.context_params = nn.Parameter(torch.zeros(latent_dim).uniform_(-latent_init_scale,latent_init_scale))
    self.latent_to_modulation = LatentToModulation(
        latent_dim=latent_dim,
        latent_vector_type=latent_vector_type,
        layer_sizes=layer_sizes,
        width=width,
        num_modulation_layers=depth-1,
        modulate_scale=modulate_scale,
        modulate_shift=modulate_shift)

    modsiren = [ModulatedSirenLayer(f_in=self.in_channels,
                                    f_out=self.width,
                                    is_first=True,
                                    w0=self.w0,
                                    modulate_scale=False,
                                    modulate_shift=False,
                                    apply_activation=False)]
    for i in range(0, self.depth-2):
      modsiren.append(ModulatedSirenLayer(f_in=self.width,
                                          f_out=self.width,
                                          w0=self.w0,
                                          modulate_scale=False,
                                          modulate_shift=False,
                                          apply_activation=False))

    modsiren.append(ModulatedSirenLayer(f_in=self.width,
                                        f_out=self.out_channels,
                                        is_last=True,
                                        w0=self.w0,
                                        modulate_scale=False,
                                        modulate_shift=False))
    
    self.modsiren = nn.ModuleList(modsiren)

  def reset_context_params(self, batch_size):
    if self.latent_vector_type == 'shared':
      rand_params = torch.zeros(self.latent_dim).uniform_(-self.latent_init_scale,self.latent_init_scale).cuda()
    elif self.latent_vector_type == 'instance':
      rand_params = torch.zeros(batch_size, self.latent_dim).uniform_(-self.latent_init_scale,self.latent_init_scale).cuda()
    self.context_params = nn.Parameter(rand_params)
    self.context_params.requires_grad = True

  def modulate(self, x: torch.Tensor, modulations: Dict[str,torch.Tensor]) -> torch.Tensor:
    """Modulates input according to modulations.

    Args:
      x: Hidden features of MLP.
      modulations: Dict with keys 'scale' and 'shift' (or only one of them)
        containing modulations.

    Returns:
      Modulated vector.
    """
    if 'scale' in modulations:
      x = modulations['scale'] * x
    if 'shift' in modulations:
      x = x + modulations['shift']
    return x

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluates model at a batch of coordinates.

    Args:
      coords (Tensor): Tensor of coordinates. Should have shape (height, width, 2)
        for images and (depth/time, height, width, 3) for 3D shapes/videos.

    Returns:
      Output features at coords.
    """
    # Compute modulations based on latent vector
    modulations = self.latent_to_modulation(self.context_params)

    # Flatten coordinates
    coords = inputs['coords']

    if self.latent_vector_type == 'shared':
      x = coords.view(-1, coords.shape[-1])
    else:
      x = coords

    # Initial layer (note all modulations are set to False here, since we
    # x - self.modsiren(x)
    # x = self.modulate(x, modulations[0])
    # x = torch.sin(self.w0 * x)

    # Hidden layers
    for i in range(0, self.depth-1):
      x = self.modsiren[i](x)
      x = self.modulate(x, modulations[i])
      x = torch.sin(self.w0 * x)

    # Final layer
    out = self.modsiren[-1](x)

    if self.latent_vector_type == 'shared':
      out = out.view(coords.size(0), coords.size(1), self.out_channels) # Unflatten output

    return {'model_in': input,
            'model_out': out, 
            'latent_vec': self.context_params}



class AsymmetricAutoEncoder(nn.Module):
  """SIREN model with FiLM modulations generated from a pre-trained encoder"""

  def __init__(self,
               width: int = 256,
               depth: int = 20,
               out_channels: int = 3,
               pretrained_encoder: str = 'resnet18',
               latent_dim: int = 512*2*2,
               latent_vector_type: str = 'instance',
               layer_sizes: Tuple[int, ...] = (512, 512),
               w0: float = 30.,
               modulate_scale: bool = False,
               modulate_shift: bool = True):
    """Constructor.

    Args:
      width (int): Width of each hidden layer in MLP.
      depth (int): Number of layers in MLP.
      out_channels (int): Number of output channels.
      latent_dim: Dimension of latent vector (input of LatentToModulation
        network).
      layer_sizes: List of hidden layer sizes for MLP parameterizing the map
        from latent to modulations. Input dimension is inferred from latent_dim
        and output dimension is inferred from number of modulations.
      w0 (float): Scale factor in sine activation in first layer.
      modulate_scale: If True, modulates scales.
      modulate_shift: If True, modulates shifts.
    """
    super().__init__()
    self.width = width
    self.depth = depth
    self.in_channels = 2
    self.out_channels = out_channels
    self.latent_dim = latent_dim
    self.layer_sizes = layer_sizes
    self.w0 = w0
    self.modulate_scale = modulate_scale
    self.modulate_shift = modulate_shift
    self.latent_vector_type = latent_vector_type

    # Initialize pre-trained model and map from latents to modulations
    #resnet = torchvision.models.resnet50(pretrained=True)
    resnet = torchvision.models.resnet18(pretrained=True)
    modules=list(resnet.children())[:-2]
    self.conv_encoder = nn.Sequential(*modules)  

    self.latent_to_modulation = LatentToModulation(
        latent_dim=latent_dim,
        latent_vector_type=latent_vector_type,
        layer_sizes=layer_sizes,
        width=width,
        num_modulation_layers=depth-1,
        modulate_scale=modulate_scale,
        modulate_shift=modulate_shift)

    modsiren = [ModulatedSirenLayer(f_in=self.in_channels,
                                    f_out=self.width,
                                    is_first=True,
                                    w0=self.w0,
                                    modulate_scale=False,
                                    modulate_shift=False,
                                    apply_activation=False)]
    for i in range(0, self.depth-2):
      modsiren.append(ModulatedSirenLayer(f_in=self.width,
                                          f_out=self.width,
                                          w0=self.w0,
                                          modulate_scale=False,
                                          modulate_shift=False,
                                          apply_activation=False))

    modsiren.append(ModulatedSirenLayer(f_in=self.width,
                                        f_out=self.out_channels,
                                        is_last=True,
                                        w0=self.w0,
                                        modulate_scale=False,
                                        modulate_shift=False))
    
    self.modsiren = nn.ModuleList(modsiren)

  def reset_context_params(self, batch_size):
    # do nothing
    a = 1

  def modulate(self, x: torch.Tensor, modulations: Dict[str,torch.Tensor]) -> torch.Tensor:
    """Modulates input according to modulations.

    Args:
      x: Hidden features of MLP.
      modulations: Dict with keys 'scale' and 'shift' (or only one of them)
        containing modulations.

    Returns:
      Modulated vector.
    """
    if 'scale' in modulations:
      x = modulations['scale'] * x
    if 'shift' in modulations:
      x = x + modulations['shift']
    return x

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """Evaluates model at a batch of coordinates.

    Args:
      coords (Tensor): Tensor of coordinates. Should have shape (height, width, 2)
        for images and (depth/time, height, width, 3) for 3D shapes/videos.

    Returns:
      Output features at coords.
    """

    encoded_feat = self.conv_encoder(inputs['spatial_img'])
    encoded_feat = encoded_feat.view(encoded_feat.size(0), -1)
  
    # Compute modulations based on latent vector
    modulations = self.latent_to_modulation(encoded_feat)

    # Flatten coordinates
    coords = inputs['coords']

    if self.latent_vector_type == 'shared':
      x = coords.view(-1, coords.shape[-1])
    else:
      x = coords

    # MLP layers
    for i in range(0, self.depth-1):
      x = self.modsiren[i](x)
      x = self.modulate(x, modulations[i])
      x = torch.sin(self.w0 * x)

    # Final layer
    out = self.modsiren[-1](x)

    if self.latent_vector_type == 'shared':
      out = out.view(coords.size(0), coords.size(1), self.out_channels) # Unflatten output

    return {'model_in': input,
            'model_out': out, 
            'latent_vec': encoded_feat}
  
  def get_parameters(self):
    return [i for i in self.latent_to_modulation.parameters()] + [i for i in self.modsiren.parameters()]

############################
# Initialization schemes
def hyper_weight_init(m, in_features_main_net):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        with torch.no_grad():
            m.bias.uniform_(-1/in_features_main_net, 1/in_features_main_net)


def hyper_bias_init(m):
    if hasattr(m, 'weight'):
        nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        m.weight.data = m.weight.data / 1.e2

    if hasattr(m, 'bias'):
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
        with torch.no_grad():
            m.bias.uniform_(-1/fan_in, 1/fan_in)
