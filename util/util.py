import torch
import cv2
import numpy as np  

def attentionmap(att_mat, original_img):
    att_mat = torch.mean(att_mat, dim=1)
    im = original_img

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n],
                                           joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()

    mask = cv2.resize(mask / mask.max(), (original_img.shape[1], original_img.shape[0]))[..., np.newaxis]
    result = (mask * im).astype("uint8")
    attention_map_1 = result
    return attention_map_1, mask

