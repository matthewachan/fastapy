"""
This script shows how to use FASTA to solve regularized least-square problem:
        min  .5||Ax-b||^2 + mu*|x|
Where A is an MxN matrix, b is an Mx1 vector of measurements, and x is the Nx1 vector of unknowns.
The parameter 'mu' controls the strength of the regularizer.

@author: Proloy DAS

"""

import numpy as np
from scipy import linalg
from fastapy import Fasta
import matplotlib.pyplot as plt
import time

import torch
from torchvision.transforms.functional import crop
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision import transforms
import torchvision
import glob
from PIL import Image

KERNEL_SIZE = 21  # approximate resolution of the PSF images
N_DEPTHS = 3  # number of point sources to include in the image (each has its own depth)
SIGMA_NOISE = 5.0 / 255  # standard deviation of the noise n
SIGMA_NLL = 0.1  # standard deviation used to computed the negative log likelihood
TAU = 0.1  # event camera thresholding value


def shrink(x, mu):
    """
    Soft theresholding function
    mu = threshold
    """
    return np.multiply(np.sign(x), np.maximum(np.abs(x) - mu, 0))


def to_event(rgb, TAU):
    """Applies event camera thresholding to an RGB frame"""
    event = torch.zeros_like(rgb[:, :])
    event[torch.abs(rgb[:, :]) <= TAU] = 0
    event[rgb[:, :] < -TAU] = -1
    event[rgb[:, :] > TAU] = 1
    return event


def load_psf(debugging=False):
    """Loads DH-PSF calibrated images at different depths"""

    def normalize(img):
        return img / torch.sum(img)

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(KERNEL_SIZE),
            transforms.Lambda(normalize),
        ]
    )

    iopen = lambda img: transform(np.array(Image.open(img)).astype(np.float32))[0]
    A = torch.tensor([], dtype=torch.float32)
    for file in glob.glob("./dh_psf/*.png"):
        psf = iopen(file).unsqueeze(0).unsqueeze(1)
        A = torch.cat([A, psf])

    # """ TODO(mchan): This line takes a subset of the DH-PSF images such that the loaded PSFs are noticeably different from one another (i.e. one DH-PSF is at ~0º orientation, another is at ~90º orientation). This makes it easier for FASTA to discern different depths, but may not reflect the true PSF wrt depth."""
    A = A[::8]

    if debugging:
        plt.title("Double helix PSFs used (A)")
        plt.imshow(torchvision.utils.make_grid(A, normalize=True).permute(1, 2, 0))
        plt.show()

    return A


def get_x(debugging=False):
    """Create point source image"""
    stdev = 1
    xx = torch.linspace(-100, 100, 256)
    xx, yy = torch.meshgrid(xx, xx, indexing="ij")
    xx = -xx

    def gauss(mean_x, mean_y, stdev):
        """Generates a Gaussian blur kernel (which is used to simulate a point source)."""
        return torch.exp(
            (-((xx - mean_x) ** 2) - (yy - mean_y) ** 2) / (2 * stdev**2)
        )

    pts = torch.linspace(-100, 100, N_DEPTHS + 1)[:-1]

    offset = torch.div(pts[1] - pts[0], 2, rounding_mode="floor")
    # x* is the ground truth scene that we'd like to recover in this inverse problem.
    x_star = torch.zeros_like(xx).unsqueeze(0).repeat([N_DEPTHS, 1, 1])
    for i, pt in enumerate(pts):
        x_star[i] = gauss(pt + offset, pt + offset, stdev)

    if debugging:
        plt.title(
            "Original scene (x).\n(Note: Each point source is at a different depth.\nBottom left is closest, top right is farthest.)"
        )
        plt.imshow(torch.sum(x_star, axis=0))
        plt.show()

    return x_star


def get_y(A, x_star, debugging=False):
    """Simulates an event camera measurement, given a scene x and a PSF A. The observation is Q(Ax+n) where Q is the event camera thresholding oeprator and n is noise."""
    Ax_star = torch.zeros_like(x_star)
    for i in torch.arange(N_DEPTHS):
        Ax_star[i] += F.conv2d(
            x_star[i].unsqueeze(0),
            A[i].unsqueeze(0),
            stride=1,
            padding=len(A[i].squeeze()) // 2,
        ).squeeze()

    Ax_star = torch.sum(Ax_star, axis=0)
    n = torch.normal(0, SIGMA_NOISE, Ax_star.size())
    y = to_event(Ax_star + n, TAU)

    if debugging:
        plt.title("Event frame Q(Ax+n)")
        plt.imshow(y)
        plt.show()

    return y


def nll(x, A, y):
    """Computes negative log likelihood of x"""
    cdf = lambda value: 0.5 * (
        1
        + torch.special.erf(
            (value) * torch.tensor(SIGMA_NLL).reciprocal() / torch.sqrt(torch.tensor(2))
        )
    )
    # Convolves x with depth-varying PSF
    Ax = torch.zeros_like(x)
    for i in torch.arange(N_DEPTHS):
        Ax[i] = F.conv2d(
            x[i].unsqueeze(0),
            A[i].unsqueeze(0),
            stride=1,
            padding=len(A[i].squeeze()) // 2,
        )
    Ax = torch.sum(Ax, axis=0)

    class1 = torch.where(y == 1, 1, 0)
    class2 = torch.where(y == 0, 1, 0)
    class3 = torch.where(y == -1, 1, 0)

    p1 = 1 - cdf(TAU - Ax)
    p2 = cdf(TAU - Ax) - cdf(-TAU - Ax)
    p3 = cdf(-TAU - Ax)

    nll = torch.tensor([0.0])
    nll += torch.sum(-class1 * torch.log(p1))
    nll += torch.sum(-class2 * torch.log(p2))
    nll += torch.sum(-class3 * torch.log(p3))

    return nll


def f(x, A, y):
    Ax = torch.zeros_like(x)
    for i in torch.arange(N_DEPTHS):
        Ax[i] = F.conv2d(
            x[i].unsqueeze(0),
            A[i].unsqueeze(0),
            stride=1,
            padding=len(A[i].squeeze()) // 2,
        )
    Ax = torch.sum(Ax, axis=0)
    return (0.5 * torch.norm(to_event(y - Ax, TAU)) ** 2).unsqueeze(0)


def debias_lasso(x, y):
    mask = torch.where(y != 0, 1, 0)
    # norm = 0
    # for xn in x:
    #     norm += torch.norm(xn * mask, p=1)
    # return norm
    return torch.norm(torch.sum(x, dim=0) * mask, p=1)


def prox_debias_lasso(x, y, mu):
    mask = torch.where(y != 0, 1, 0)
    return mask * torch.mul(
        torch.sign(x), torch.maximum(torch.abs(x) - mu, torch.tensor([0]))
    )


def gd(A, x0, x_star, y, max_iter=100, lr=1e-2, mu=1):
    x = x0
    best_x = x
    err = [torch.norm(x.detach() - x_star)]

    for i in torch.arange(max_iter):
        if i % 50:
            lr /= 2

        obj = nll(x, A, y) + mu * debias_lasso(x, y)
        obj.backward(gradient=torch.tensor([1.0]))

        # Gradient descent
        x = (x - lr * x.grad).detach()
        err.append(torch.norm(x.detach() - x_star))
        if err[-1] < np.amin(err[:-1]):
            best_x = x
        x.requires_grad_(True)

    return (x.detach(), best_x.detach(), err)


def invert(x_star, A, y, debugging=False):
    x0 = torch.zeros_like(x_star).requires_grad_(True)
    x, best_x, err = gd(A, x0, x_star, y, max_iter=100, lr=1e-2, mu=20)

    if debugging:
        print(f"Min residual: {np.amin(err)}, last residual: {err[-1]}")
        plt.title("Reconstructed x at different depths")
        plt.imshow(make_grid(best_x.unsqueeze(1), pad_value=1, normalize=True)[0])
        plt.show()
        plt.title("Observation, sum of multi-depth x's, and ground truth")
        plt.imshow(
            make_grid(
                torch.cat(
                    [
                        y.unsqueeze(0),
                        torch.sum(best_x, axis=0, keepdim=True),
                        torch.sum(x_star, axis=0, keepdim=True),
                    ]
                ).unsqueeze(1),
                pad_value=1,
                normalize=True,
                scale_each=True,
            )[0]
        )
        plt.show()

    return best_x


def setup_rls(debugging=False):
    mu = 10  # regularization parameter

    # Create sparse signal
    x = get_x(debugging)

    # define random Gaussian matrix
    A = load_psf(debugging)

    # Define observation vector
    b = get_y(A, x, debugging)

    #  The initial iterate:  a guess at the solution
    x0 = torch.zeros_like(x)

    # Create function handles
    def f(x):
        return nll(x, A, b)  # Negative log likelihood

    def gradf(x):
        xn = x.detach().clone().requires_grad_(True)
        loss = f(xn)
        loss.backward(gradient=torch.tensor([1.0]))
        return xn.grad  # gradient of f(x) via autograd

    def g(x):
        return mu * debias_lasso(x, b)
        # return mu * torch.norm(x, p=1)  # mu*|x|

    def proxg(x, t):
        # return shrink(x, mu * t)  # proximal operator for g(x)
        return prox_debias_lasso(x, b, mu * t)

    return f, gradf, g, proxg, x0, x


def test_rls(debugging=False):
    f, gradf, g, proxg, x0, x = setup_rls(debugging)
    # Set up Fasta solver
    lsq = Fasta(f, g, gradf, proxg)
    # Call Solver
    lsq.learn(x0, verbose=True)

    if debugging:
        print(lsq.residuals[0], lsq.residuals[-1], np.amin(lsq.residuals))
        print(lsq.residuals[-1] / lsq.residuals[0])
        x_hat = torch.sum(lsq.coefs_, dim=0)
        plt.title("Recovered scene x_hat (FASTA)")
        plt.imshow(x_hat)
        plt.show()
        plt.title("Residual ||x*-x_hat||")
        # print(x_hat.size(), x.size())
        plt.imshow(torch.sqrt((torch.sum(lsq.coefs_, dim=0) - x[1]) ** 2))
        plt.show()
        for i in range(len(lsq.coefs_)):
            plt.title(f"Recovered scene at depth z={i}")
            plt.imshow(lsq.coefs_[i])
            plt.show()
    # assert lsq.residuals[-1] / lsq.residuals[0] < 1e-4

    # if debugging:
    #     plt.figure("sparse least-square")
    #     plt.subplot(2, 1, 1)
    #     plt.stem(x, markerfmt="go", linefmt="g:", label="Ground truth")
    #     plt.stem(lsq.coefs_, markerfmt="bo", label="Fasta solution")
    #     plt.xlabel("Index")
    #     plt.ylabel("Signal Value")

    #     plt.subplot(2, 1, 2)
    #     plt.semilogy(lsq.residuals)

    #     plt.show()


if __name__ == "__main__":
    test_rls(True)
    # h = load_psf()
    # x = get_x()
    # y = get_y(h, x)
    # invert(x, h, y, True)
