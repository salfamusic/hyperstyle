from module.flow import cnf
import torch
import tensorflow as tf
import numpy as np


def edit_attribute(w_latents, attributes, lighting, session, model, w_avg, flow_model, direction, strength):
    w_latents = w_latents.cpu().data.numpy()
    attribute_names = ['Gender', 'Glasses', 'Yaw', 'Pitch', 'Baldness', 'Beard', 'Age', 'Expression']
    attr_degree_list = [1.5, 2.5, 1., 1., 2, 1.7, 0.93, 1.]

    light_names = ['Left->Right', 'Right->Left', 'Down->Up', 'Up->Down', 'No light', 'Front light']

    att_min = {'Gender': -5, 'Glasses': -5, 'Yaw': -100, 'Pitch': -100, 'Baldness': -5, 'Beard': -5.0, 'Age': 0,
               'Expression': -5}
    att_max = {'Gender': 5, 'Glasses': 5, 'Yaw': 100, 'Pitch': 100, 'Baldness': 5, 'Beard': 5, 'Age': 65, 'Expression': 5}
    z_latents = flow_w_to_z(flow_model, w_latents, attributes.ravel(), lighting)
    new_w_latents = w_latents

    att_new = list(attributes)
    attr_idx = -1

    for i, att in enumerate(attribute_names):  # Not the greatest code, but works!

        if att != direction:
            continue
        attr_idx = i
        attr_change = strength - attributes[i]

        attr_final = attr_degree_list[i] * attr_change + attributes[i]
        att_new[i] = attr_final
        new_w_latents = flow_z_to_w(flow_model, z_latents, np.array(att_new).ravel(), lighting)
        break

    if attr_idx == -1:
        return w_latents

    return torch.Tensor(preserve_w_id(new_w_latents, w_latents, attr_idx)).cuda(), lighting, att_new

    

def np_copy(*args):  # shortcut to clone multiple arrays
    return [np.copy(arg) for arg in args]


def preserve_w_id(w_new, w_orig, attr_index):
    # Ssssh! secret sauce to strip vectors
    w_orig = torch.Tensor(w_orig)
    if attr_index == 0:
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 1:
        w_new[0][:2] = w_orig[0][:2]
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 2:

        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 3:
        w_new[0][4:] = w_orig[0][4:]

    elif attr_index == 4:
        w_new[0][6:] = w_orig[0][6:]

    elif attr_index == 5:
        w_new[0][:5] = w_orig[0][:5]
        w_new[0][10:] = w_orig[0][10:]

    elif attr_index == 6:
        w_new[0][0:4] = w_orig[0][0:4]
        w_new[0][8:] = w_orig[0][8:]

    elif attr_index == 7:
        w_new[0][:4] = w_orig[0][:4]
        w_new[0][6:] = w_orig[0][6:]
    return w_new

def flow_w_to_z(flow_model, w, attributes, lighting):
    w_cuda = torch.Tensor(w)
    att_cuda = torch.from_numpy(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    z = flow_model(w_cuda, features, zero_padding)[0].clone().detach()

    return z

def flow_z_to_w(flow_model, z, attributes, lighting):
    att_cuda = torch.Tensor(np.asarray(attributes)).float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    light_cuda = torch.Tensor(lighting)

    features = torch.cat([light_cuda, att_cuda], dim=1).clone().detach()
    zero_padding = torch.zeros(1, 18, 1)
    w = flow_model(z, features, zero_padding, True)[0].clone().detach().numpy()

    return w
