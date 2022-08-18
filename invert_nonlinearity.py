#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.transforms.functional import crop
import torch
import numpy as np

# import skimage
import matplotlib.pyplot as plt

# from torch.distributions.normal import Normal
# from scipy import signal
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision

# import cv2
import glob
from PIL import Image


# In[2]:


plt.rcParams["figure.figsize"] = [10, 5]


def to_event(rgb, tau):
    event = torch.zeros_like(rgb[:, :])
    event[torch.abs(rgb[:, :]) <= tau] = 0
    event[rgb[:, :] < -tau] = -1
    event[rgb[:, :] > tau] = 1
    return event


def crop(img):
    return transforms.functional.crop(img, 0, 5, 39, 39)


def normalize(img):
    return img / torch.sum(img)


# # Setup
# Compose a scene with four point sources at different depths, convolved with a double-helix PSF.

# In[3]:


""" Load PSFs """
kernel_size = 21
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.CenterCrop(kernel_size),
        transforms.Lambda(normalize),
    ]
)

iopen = lambda img: transform(np.array(Image.open(img)).astype(np.float32))[0]
A = torch.tensor([], dtype=torch.float32)
for file in glob.glob("./dh_psf/*.png"):
    psf = iopen(file).unsqueeze(0).unsqueeze(1)
    A = torch.cat([A, psf])

# plt.title("Double helix PSFs")
# plt.imshow(torchvision.utils.make_grid(A, normalize=True).permute(1, 2, 0))
# plt.show()
""" TODO(mchan): Remove this"""
A = A[::8]
# A = A[8].unsqueeze(0).repeat([4, 1, 1, 1])

""" Create point source image """
stdev = 1
xx = torch.linspace(-100, 100, 256)
xx, yy = torch.meshgrid(xx, xx)
xx = -xx


def gauss(mean_x, mean_y, stdev):
    return torch.exp((-((xx - mean_x) ** 2) - (yy - mean_y) ** 2) / (2 * stdev ** 2))


pts = torch.linspace(-100, 100, len(A) + 1)[:-1]

offset = torch.div(pts[1] - pts[0], 2, rounding_mode="floor")
x_star = torch.zeros_like(xx).unsqueeze(0).repeat([len(A), 1, 1])
for i, pt in enumerate(pts):
    x_star[i] = gauss(pt + offset, pt + offset, stdev)
# x_star -= torch.roll(x_star, (0, 5), (0, 1))
# plt.title("Original scene (x)")
# plt.imshow(torch.sum(x_star, axis=0))
# plt.show()


# In[4]:


Ax_star = torch.zeros_like(x_star)
for i in torch.arange(len(A)):
    Ax_star[i] += F.conv2d(
        x_star[i].unsqueeze(0),
        A[i].unsqueeze(0),
        stride=1,
        padding=len(A[i].squeeze()) // 2,
    ).squeeze()

# plt.title("Convolved scene (Ax)")
# plt.imshow(torch.sum(Ax_star, axis=0))
# plt.show()
print(torch.amax(x_star), torch.amax(Ax_star))


# # Add nonlinearity
# Create the event camera image by thresholding to ${-1, 0, 1}$ depending on log intensity.

# In[5]:


tau = 0.1
sigma = 5 / 255
print(f"sigma val: {sigma}")

n = torch.normal(0, sigma, xx.shape)
b = to_event(torch.sum(Ax_star, axis=0) + n, tau)
# plt.title("Event frame Q(Ax+n)")
# plt.imshow(b)
# plt.show()


# # Proof of concept
# Verify that relative (ordinal) depth can be recovered if $Q^{-1}(y)$ (without noise) and $x^*$ are perfectly recovered.

# In[8]:


# n_depths = len(A) + 1  # Extra depth layer for the background
# """ Compute depth segmentation map (assuming perfect knowledge) """
# Qinv_y = torch.sum(Ax_star, axis=0).unsqueeze(0).repeat([n_depths, 1, 1])

# Ax = torch.sum(x_star, axis=0).unsqueeze(0)
# for i in torch.arange(n_depths - 1):
#     tmp = F.conv2d(
#         torch.sum(x_star, axis=0).unsqueeze(0),
#         A[i].unsqueeze(0),
#         stride=1,
#         padding=kernel_size // 2,
#     )
#     Ax = torch.cat([Ax, tmp])

# plt.title("Image convolved with different kernels")
# plt.imshow(torchvision.utils.make_grid(Ax.unsqueeze(1), normalize=True)[0])
# plt.show()

# # Finds the minimum energy slice.
# e = (Qinv_y - Ax) ** 2
# depth = np.argmin(e, axis=0)

# seg_mask = depth[:, :]
# segmap = np.zeros_like(seg_mask).astype(np.float64)
# for i in np.arange(n_depths):
#     segmap += np.where(seg_mask == i, i, 0)
# plt.title("Depth segmentation map")
# plt.imshow(segmap)
# plt.show()


# # Inverse imaging
# Invert the $A$ without the non-linearity applied.
#
# $$\frac{1}{2}\|\sum b - Ax\|^2_2 $$
# $$\nabla_{x_1} = A^T(b - Ax) = 0 $$
# $$ A^Tb = A^TAx $$
# $$ x = (A^TA)^{-1}A^Tb $$

# In[9]:


# def f(A, x, b):
#     Ax = torch.zeros_like(x)
#     for i in torch.arange(len(A)):
#         Ax[i] = F.conv2d(
#             x[i].unsqueeze(0),
#             A[i].unsqueeze(0),
#             stride=1,
#             padding=len(A[i].squeeze()) // 2,
#         )
#     Ax = torch.sum(Ax, axis=0)

#     return (0.5 * torch.norm(Ax - b)).unsqueeze(0)


# x0 = torch.zeros_like(x_star).requires_grad_(True)
# lr = 1e-2

# # Gradient descent loop
# err = [torch.norm(x0.detach() - x_star)]
# for i in torch.arange(200):
#     if i % 50:
#         lr /= 2
#     obj = f(A, x0, torch.sum(Ax_star, axis=0)) + 0.2 * torch.norm(x0, p=1)
#     obj.backward(gradient=torch.tensor([1.0]))
#     x0 = (x0 - lr * x0.grad).detach()
#     err.append(torch.norm(x0 - x_star))
#     x0 = x0.requires_grad_(True)
# print(np.amin(err), err[-1])
# plt.title("Multi-depth x reconstructions")
# plt.imshow(make_grid(x0.unsqueeze(1), normalize=True, scale_each=True)[0])


# In[10]:


# x0 = torch.zeros_like(x_star).unsqueeze(0)
# y0 = x0.detach().clone()
# l0 = x0.detach().clone()
# lr = 1e-2
# tau = 1e-3
# lamb = 1e-3

# # ADMM loop
# err = [torch.norm(x0.detach() - x_star)]
# b = torch.sum(Ax_star, axis=0).unsqueeze(0)

# # Precompute stuff
# AtA = torch.zeros_like(A)
# for i in torch.arange(len(A)):
#     AtA[i] = F.conv2d(A[i].transpose(1, 2).unsqueeze(0), A[i].unsqueeze(0), stride=1, padding=len(A[i].squeeze())//2)
#     AtA[i] = torch.inverse(AtA[i] + tau * torch.eye(A.size()[-1]))
# offset = (Ax_star.size()[-1] - AtA.size()[-1]) // 2
# AtA = F.pad(AtA, (offset, offset+1, offset, offset+1, 0, 0))
# AtA = AtA.squeeze()
# Atb = torch.zeros_like(x_star)
# for i in torch.arange(len(A)):
#     Atb[i] = F.conv2d(b.unsqueeze(0), A[i].transpose(1, 2).unsqueeze(0), stride=1, padding=len(A[i].squeeze())//2)
# # print(AtA.size(), Atb.size(), Ax_star.size(), A.size())

# for i in torch.arange(1):
#     x0 = AtA @ (Atb + tau * y0 - tau * l0)
#     y0 = x0 - l0 - torch.maximum(torch.minimum(tau * torch.eye(x0.size()[-1]), x0 - l0), -tau * torch.eye(x0.size()[-1]))
#     l0 += x0 - y0
# x0 = x0.squeeze(0)
# #     err.append(torch.norm(x0 - x_star))
# #     x0 = x0.requires_grad_(True)
# # print(np.amin(err), err[-1])
# plt.title('Multi-depth x reconstructions')
# plt.imshow(make_grid(x0.unsqueeze(1), normalize=True, scale_each=True)[0])


# In[11]:


# plt.title('')
# plt.imshow(make_grid(x0.unsqueeze(1), pad_value=1, normalize=True)[0])
# plt.title("Observation, sum of multi-depth x's, and ground truth")
# plt.imshow(
#     make_grid(
#         torch.cat(
#             [
#                 torch.sum(Ax_star, axis=0, keepdim=True),
#                 torch.sum(x0, axis=0, keepdim=True),
#                 torch.sum(x_star, axis=0, keepdim=True),
#             ]
#         ).unsqueeze(1),
#         normalize=True,
#         scale_each=True,
#     )[0]
# )
# plt.imshow(torch.sum(x0, axis=0).detach())


# # Negative log-likelihood
# Forward pass is given by:
# $$ -log(p(y)) = -\sum_{i,y_i=1}log(p(y_i=1))-\sum_{i,y_i=0}log(p(y_i=0))-\sum_{i,y_i=-1}log(p(y_i=-1)) $$
#

# In[12]:


def nll(x, A, b, sigma, tau):
    print(f"sigma nll {sigma}")
    cdf = lambda value: 0.5 * (
        1
        + torch.special.erf(
            (value) * torch.tensor(sigma).reciprocal() / torch.sqrt(torch.tensor(2))
        )
    )
    # Convolves x with depth-varying PSF
    Ax = torch.zeros_like(x)
    for i in torch.arange(len(A)):
        Ax[i] = F.conv2d(
            x[i].unsqueeze(0),
            A[i].unsqueeze(0),
            stride=1,
            padding=len(A[i].squeeze()) // 2,
        )
    Ax = torch.sum(Ax, axis=0)

    class1 = torch.where(b == 1, 1, 0)
    class2 = torch.where(b == 0, 1, 0)
    class3 = torch.where(b == -1, 1, 0)

    p1 = 1 - cdf(tau - Ax)
    p2 = cdf(tau - Ax) - cdf(-tau - Ax)
    p3 = cdf(-tau - Ax)

    nll = torch.tensor([0.0])
    nll += torch.sum(-class1 * torch.log(p1))
    nll += torch.sum(-class2 * torch.log(p2))
    nll += torch.sum(-class3 * torch.log(p3))
    print(f"sigma {sigma}, tau {tau}, nll {nll}")

    return nll


# In[13]:


def f(x, A, b, sigma, tau):
    Ax = torch.zeros_like(x)
    for i in torch.arange(len(A)):
        Ax[i] = F.conv2d(
            x[i].unsqueeze(0),
            A[i].unsqueeze(0),
            stride=1,
            padding=len(A[i].squeeze()) // 2,
        )
    Ax = torch.sum(Ax, axis=0)
    return (0.5 * torch.norm(to_event(b - Ax, tau)) ** 2).unsqueeze(0)


# # Objective
# $$ \underset{y}{argmin} -log(p(y)) + \lambda|y| $$

# In[28]:


""" Only apply LASSO penalty on pixels where the observation is non-zero"""


def debias_lasso(x, b):
    mask = torch.where(b != 0, 1, 0)
    norm = 0
    for xn in x:
        norm += torch.norm(xn * mask, p=1)
    print(f"debias {norm}")
    return norm
    # return torch.norm(torch.sum(x, dim=0) * mask, p=1)


def invert(A, x0, b, sigma, max_iter=5, lr=1e-2, weight=0.4):
    x = x0
    best_x = x
    err = [torch.norm(x.detach() - x_star)]

    for i in torch.arange(max_iter):
        if i % 50:
            lr /= 2

        obj = nll(x, A, b, sigma, tau) + weight * debias_lasso(x, b)
        print(obj)
        obj.backward(gradient=torch.tensor([1.0]))

        # Gradient descent
        x = (x - lr * x.grad).detach()
        err.append(torch.norm(x.detach() - x_star))
        if err[-1] < np.amin(err[:-1]):
            best_x = x
        x.requires_grad_(True)

    return (x.detach(), best_x.detach(), err)


# In[30]:


""" Joint optimization (with alpha mask) """
x0 = torch.zeros_like(x_star).requires_grad_(True)
print(torch.mean(A), torch.mean(x_star), torch.mean(b))
x, best_x, err = invert(A, x0, b, 0.1, max_iter=10, lr=1e-2, weight=10)
print(np.amin(err), err[-1])

# plt.rcParams['figure.figsize'] = [20, 10]
plt.title("Reconstructed x at different depths")
plt.imshow(make_grid(best_x.unsqueeze(1), pad_value=1, normalize=True)[0])
plt.show()
# plt.title("Observation, sum of multi-depth x's, and ground truth")
# plt.imshow(
#     make_grid(
#         torch.cat(
#             [
#                 b.unsqueeze(0),
#                 torch.sum(best_x, axis=0, keepdim=True),
#                 torch.sum(x_star, axis=0, keepdim=True),
#             ]
#         ).unsqueeze(1),
#         pad_value=1,
#         normalize=True,
#         scale_each=True,
#     )[0]
# )
# plt.show()


# In[31]:


""" Individually solve """
# x = torch.tensor([])
# m = torch.ones_like(x_star)
# for psf in A:
#     x_i, best_x_i, residual = invert(psf.unsqueeze(0), m)
#     x = torch.cat([x, best_x_i.detach().unsqueeze(0)])

# plt.imshow(b)
# plt.show()
# plt.imshow(torchvision.utils.make_grid(x.unsqueeze(1), normalize=True)[0])
# plt.show()
# plt.imshow(torchvision.utils.make_grid(A, normalize=True)[0])
# plt.show()


# In[17]:


""" Joint solve (ignoring masking) """
# m = torch.ones_like(x_star).repeat([len(A), 1, 1])
# x_i, best_x_i, residual = invert(A, m)
# plt.imshow(x_i.detach().numpy())


# # Depth recovery

# In[18]:


# Ax = torch.tensor([], dtype=torch.float32)
# for i in torch.arange(len(A)):
#     blur = F.conv2d(x_i.unsqueeze(0), A[i].unsqueeze(0), stride=1, padding=len(A[i].squeeze())//2)
#     Ax = torch.cat((Ax, blur), axis=0)
# # plt.imshow(torch.sum(Ax, axis=0).detach())
# # total = torch.sum(Ax, axis=0).repeat([len(A), 1, 1])
# total = Ax_star.repeat([len(A), 1, 1])
# e = (total - Ax)**2
# # e = (total - Ax)**2
# depth = np.argmin(e, axis=0)

# seg_mask = depth[:,:]
# segmap = np.zeros_like(seg_mask).astype(np.float64)
# for i in np.arange(n_depths):
#     segmap += np.where(seg_mask == i, i, 0)
# plt.title('Depth segmentation map')
# plt.imshow(segmap)
