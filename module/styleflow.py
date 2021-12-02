from utils.sf_utils import Build_model
from module.flow import cnf
import tensorflow as tf
import torch

def get_styleflow_model():
    # Open a new TensorFlow session.
    config = tf.ConfigProto(allow_soft_placement=True)
    session = tf.Session(config=config)

    with session.as_default():
        model = Build_model('gdrive:networks/stylegan2-ffhq-config-f.pkl')
        w_avg = model.Gs.get_var('dlatent_avg')

    prior = cnf(512, '512-512-512-512-512', 17, 1)
    prior.load_state_dict(torch.load('flow_weight/modellarge10k.pt'))
    prior.eval()

    return session, model, w_avg, prior.cpu()