# %matplotlib inline

import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim


def build_cov_matrix(rho):
    return torch.tensor([
        [1, 0, 0, rho],
        [0, 1, rho, 0],
        [0, rho, 1, 0],
        [rho, 0, 0, 1]]
    ).float()


lr = 0.001
last_lr = 0.000001
batch_size = 128
total_iters = 2000
display_step = 400


def mi_theta(T, pos, neg):
    return T(pos).mean() - T(neg).exp().mean().log()


# rhos = np.linspace(-0.999, 0.999, num=13)
rhos = [-0.99, -0.90, -0.70, -0.50, -0.30, -0.10, 0.00, 0.10, 0.30, 0.50, 0.70, 0.90, 0.99]
mis = []  # mutual_information
for rho in rhos:
    cov_matrix = build_cov_matrix(rho)
    X_ab = dist.MultivariateNormal(torch.zeros(4), cov_matrix)
    X_b = dist.MultivariateNormal(torch.zeros(2), cov_matrix[2:, 2:])

    # network
    T = nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    plot_loss = []
    trainer = optim.SGD(T.parameters(), lr=lr, momentum=0.9)

    print('==================rho:%.2f=================' % rho)
    for it in range(total_iters):
        xab_samples = X_ab.sample((batch_size,))
        xb_samples = X_b.sample((batch_size,))
        neg_samples = torch.cat([xab_samples[:, :2], xb_samples], -1)

        loss = - mi_theta(T, xab_samples, neg_samples)
        trainer.zero_grad()
        loss.backward()
        trainer.step()
        plot_loss.append(loss.item())

        update_lr = lr - (lr - last_lr) * display_step / total_iters
        for param_group in trainer.param_groups:
            param_group['lr'] = update_lr

        if (it + 1) % display_step == 0:
            print('[Iter: %d] [loss: %.3f]' % (it, sum(plot_loss[-display_step:]) / display_step))

    xab_test_samples = X_ab.sample((1000,))
    xb_test_samples = X_b.sample((1000,))
    neg_test_samples = torch.cat([xab_test_samples[:, :2], xb_test_samples], -1)
    mi = mi_theta(T, xab_test_samples, neg_test_samples).mean().item()
    mis.append(mi)
    print('===================mi:%.2f=================' % mi)

# plot
plt.plot(rhos, mis, 'r')