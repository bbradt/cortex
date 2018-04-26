'''Simple GAN model

'''

import math

import torch
from torch import autograd
import torch.nn.functional as F

from .vae import update_decoder_args, update_encoder_args


def log_sum_exp(x, axis=None):
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def raise_measure_error(measure):
    supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
    raise NotImplementedError('Measure `{}` not supported. Supported: {}'.format(measure, supported_measures))


def get_positive_expectation(p_samples, measure):
    log_2 = math.log(2.)

    if   measure == 'GAN': Ep = -F.softplus(-p_samples)
    elif measure == 'JSD': Ep = log_2 - F.softplus(-p_samples)
    elif measure == 'X2':  Ep = p_samples ** 2
    elif measure == 'KL':  Ep = p_samples + 1.
    elif measure == 'RKL': Ep = -torch.exp(-p_samples)
    elif measure == 'DV':  Ep = p_samples
    elif measure == 'H2':  Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':  Ep = p_samples
    else:
        raise_measure_error(measure)

    return Ep.mean()


def get_negative_expectation(q_samples, measure):
    log_2 = math.log(2.)

    if   measure == 'GAN': Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD': Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':  Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':  Eq = torch.exp(q_samples)
    elif measure == 'RKL': Eq = q_samples - 1.
    elif measure == 'DV':  Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':  Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':  Eq = q_samples
    else:
        raise_measure_error(measure)

    return Eq.mean()


def get_boundary(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'H2', 'DV'):
        b =samples ** 2
    elif measure == 'X2':
        b = (samples / 2.) ** 2
    elif measure == 'W':
        b = None
    else:
        raise_measure_error(measure)

    return b.mean()

def get_weight(samples, measure):
    if measure in ('GAN', 'JSD', 'KL', 'RKL', 'DV', 'H2'):
        return samples ** 2
    elif measure == 'X2':
        return (samples / 2.) ** 2
    elif measure == 'W1':
        return None
    else:
        raise_measure_error(measure)


def generator_loss(q_samples, measure, loss_type=None):
    if not loss_type:
        return get_negative_expectation(q_samples, measure)
    elif loss_type == 'non-saturating':
        return -get_positive_expectation(q_samples, measure)
    elif loss_type == 'boundary-seek':
        return get_boundary(q_samples, measure)
    else:
        raise NotImplementedError('Generator loss type `{}` not supported. '
                                  'Supported: [None, non-saturating, boundary-seek]')


def apply_gradient_penalty(data, models, losses, results, inputs=None, model=None, penalty_type='gradient_norm',
                           penalty_amount=1.0):
    if penalty_amount == 0.:
        return
    if inputs is None:
        raise ValueError('No inputs provided')
    if not model:
        raise ValueError('No model provided')

    model_ = models[model]
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]

    inputs = [inp.detach() for inp in inputs]
    [inp.requires_grad_() for inp in inputs]

    def get_gradient(inp, output):
        gradient = autograd.grad(outputs=output, inputs=inp, grad_outputs=torch.ones_like(output),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]
        return gradient

    if penalty_type == 'gradient_norm':
        penalties = []
        for inp in inputs:
            output = model_(inp)
            gradient = get_gradient(inp, output)
            gradient = gradient.view(gradient.size()[0], -1)
            penalties.append((gradient ** 2).sum(1).mean())
        penalty = sum(penalties)

    elif penalty_type == 'interpolate':
        if len(inputs) != 2:
            raise ValueError('tuple of 2 inputs required to interpolate')
        inp1, inp2 = inputs

        try:
            epsilon = data['e'].view(-1, 1, 1, 1)
        except:
            raise ValueError('You must initiate a uniform random variable `e` to use interpolation')
        mid_in = ((1. - epsilon) * inp1 + epsilon * inp2)
        mid_in.requires_grad_()

        mid_out = model_(mid_in)
        gradient = get_gradient(mid_in, mid_out)
        gradient = gradient.view(gradient.size()[0], -1)
        penalty = ((gradient.norm(1) - 1.) ** 2).mean()

    else:
        raise NotImplementedError('Unsupported penalty {}'.format(penalty_type))

    if model in losses:
        losses[model] += penalty_amount * penalty
    else:
        losses[model] = penalty_amount * penalty

    results[model + '_loss_wo_penalty'] = losses[model].item()
    results[model + '_penalty'] = penalty.item()


def setup(routines=None, **kwargs):
    routines['generator']['measure'] = routines['discriminator']['measure']


def discriminator_routine_test(data, models, losses, results, viz, measure=None, **penalty_args):
    Z, X_P = data.get_batch('z', 'images')
    discriminator = models['discriminator']
    generator = models['generator']

    X_Q = generator(Z, nonlinearity=F.tanh)
    P_samples = discriminator(X_P)
    Q_samples = discriminator(X_Q)

    E_pos = get_positive_expectation(P_samples, measure)
    E_neg = get_negative_expectation(Q_samples, measure)
    difference = E_pos - E_neg

    losses.update(discriminator=-difference)

    results.update(Scores=dict(Ep=P_samples.mean().item(), Eq=Q_samples.mean().item()))
    results['{} distance'.format(measure)] = difference.item()
    viz.add_image(X_P, name='ground truth')
    viz.add_histogram(dict(fake=Q_samples.view(-1).data, real=P_samples.view(-1).data), name='discriminator output')

    return X_P, X_Q


def discriminator_routine(data, models, losses, results, viz, measure=None, **penalty_args):
    Z, X_P = data.get_batch('z', 'images')
    generator = models['generator']

    X_Q = generator(Z, nonlinearity=F.tanh)
    discriminator_routine_test(data, models, losses, results, viz, measure=measure)
    apply_gradient_penalty(data, models, losses, results, inputs=(X_P, X_Q), model='discriminator', **penalty_args)


def generator_routine(data, models, losses, results, viz, measure=None, loss_type=None):
    Z = data['z']
    discriminator = models['discriminator']
    generator = models['generator']

    X_Q = generator(Z, nonlinearity=F.tanh)
    samples = discriminator(X_Q)

    g_loss = generator_loss(samples, measure, loss_type=loss_type)
    weights = get_weight(samples, measure)

    losses.update(generator=g_loss)
    results.update(Weights=weights.mean().item())
    viz.add_image(X_Q, name='generated')


def build_model(data, models, model_type='dcgan', discriminator_args=None, generator_args=None):
    x_shape = data.get_dims('x', 'y', 'c')
    dim_z = data.get_dims('z')

    Encoder, discriminator_args = update_encoder_args(x_shape, model_type=model_type, encoder_args=discriminator_args)
    Decoder, generator_args = update_decoder_args(x_shape, model_type=model_type, decoder_args=generator_args)

    discriminator = Encoder(x_shape, dim_out=1, **discriminator_args)
    generator = Decoder(x_shape, dim_in=dim_z, **generator_args)

    models.update(generator=generator, discriminator=discriminator)


ROUTINES = dict(discriminator=(discriminator_routine, discriminator_routine_test), generator=generator_routine)

DEFAULT_CONFIG = dict(
    data=dict(batch_size=dict(train=64, test=1000), skip_last_batch=True,
              noise_variables=dict(z=dict(dist='normal', size=64, loc=0, scale=1),
                                   e=dict(dist='uniform', size=1, low=0, high=1))),
    optimizer=dict(optimizer='Adam', learning_rate=1e-4, updates_per_model=dict(discriminator=1, generator=1)),
    model=dict(model_type='convnet', discriminator_args=None, generator_args=None),
    routines=dict(discriminator=dict(measure='GAN', penalty_type='gradient_norm', penalty_amount=1.0),
                  generator=dict(loss_type='non-saturating')),
    train=dict(epochs=100, archive_every=10)
)