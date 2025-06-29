import torch
import numpy as np
import matplotlib.pyplot as plt

def viz_trajs_no_resample(x, lon_range, lat_range):
    """Give a trajs array and length array, visulaize the trajs
       x: (B, N, 2)
       lenght: (B,)
       spatio_mean: (2,)
       spatio_std: (2,)
       chengdu: [104.03968953679004, 104.12705400673643], [30.655400079856072, 30.730172829483855]
    """
    plt.figure(figsize=(8,8))
    for i in range(len(x)):
        traj=x[i]
        traj[:,0] = np.clip(traj[:,0], lon_range[0], lon_range[1])
        traj[:,1] = np.clip(traj[:,1], lat_range[0], lat_range[1])
        plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
    plt.axis('off')
    # plt.tight_layout()
    # save
    # plt.savefig('Chengdu_traj_3000_test.png')
    # plt.show()
    # Use `canvas` to retrieve the RGB image as a NumPy array
    plt.gcf().canvas.draw()  # Draw the canvas to update the plot
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()  # Close the figure to free memory
    return image

def resample_trajectory(x, length=200):
    """
    Resamples a trajectory to a new length.

    Parameters:
        x (np.ndarray): original trajectory, shape (N, 2)
        length (int): length of resampled trajectory

    Returns:
        np.ndarray: resampled trajectory, shape (length, 2)
    """
    len_x = len(x)
    time_steps = np.arange(length) * (len_x - 1) / (length - 1)
    x = x.T
    resampled_trajectory = np.zeros((2, length))
    for i in range(2):
        resampled_trajectory[i] = np.interp(time_steps, np.arange(len_x), x[i])
    return resampled_trajectory.T

def viz_trajs(x, lengths, lon_range, lat_range, spatio_mean = None, spatio_std = None):
    """Give a trajs array and length array, visulaize the trajs
       x: (B, N, 2)
       lenght: (B,)
       spatio_mean: (2,)
       spatio_std: (2,)
       chengdu: [104.03968953679004, 104.12705400673643], [30.655400079856072, 30.730172829483855]
    """
    Gen_traj = []
    for j in range(x.shape[0]):
        new_traj = resample_trajectory(x[j], lengths[j])
        if spatio_mean is not None and spatio_std is not None:
            new_traj = new_traj * spatio_std + spatio_mean
        new_traj[:,0] = np.clip(new_traj[:,0], lon_range[0], lon_range[1])
        new_traj[:,1] = np.clip(new_traj[:,1], lat_range[0], lat_range[1])
        assert new_traj.dtype == np.float64
        Gen_traj.append(new_traj)
    assert len(Gen_traj) == x.shape[0]

    plt.figure(figsize=(8,8))
    for i in range(len(Gen_traj)):
        traj=Gen_traj[i]
        plt.plot(traj[:,0],traj[:,1],color='blue',alpha=0.1)
    plt.axis('off')
    # plt.tight_layout()
    # save
    plt.savefig('Chengdu_traj_3000_test.png')
    # plt.show()
    # Use `canvas` to retrieve the RGB image as a NumPy array
    plt.gcf().canvas.draw()  # Draw the canvas to update the plot
    image = np.frombuffer(plt.gcf().canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(plt.gcf().canvas.get_width_height()[::-1] + (3,))
    plt.close()  # Close the figure to free memory
    return image

class DDPM:
    def __init__(self,
                 T: int,
                 noise_level_share: bool,
                 device: str,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
    ) -> None:
        ''' T is the max diffusion step. noise_level_share indicate whether add different
        noise levels among the sequence dimension(2nd dimension) '''
        self.T = T
        self.noise_level_share = noise_level_share
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float32).to(device)
        self.one_minus_betas = 1 - self.betas
        self.alphas = torch.cumprod(self.one_minus_betas, dim=0) # someplaces use alpha_bar

    def forward(self, x: torch.tensor):
        x = x.to(self.device)
        B, L = x.shape[0], x.shape[1]

        # diffuse
        noise = torch.randn_like(x)
        noise_levels = None
        if self.noise_level_share:
            noise_levels = torch.randint(0, self.T, (B,), device=self.device)
        else:
            noise_levels = torch.randint(0, self.T, (B, L), device=self.device)
        alphas_sqrt = self.alphas[noise_levels].sqrt()
        one_minus_alphas_sqrt = (1 - self.alphas[noise_levels]).sqrt()
        while alphas_sqrt.ndim < x.ndim:
            alphas_sqrt = alphas_sqrt.unsqueeze(-1)
            one_minus_alphas_sqrt = one_minus_alphas_sqrt.unsqueeze(-1)
            assert alphas_sqrt.ndim == one_minus_alphas_sqrt.ndim
        
        x_noise = alphas_sqrt * x + one_minus_alphas_sqrt * noise
        return x_noise, noise, noise_levels
    
    def forward_with_noise_level(self, x: torch.tensor, noise_levels: torch.tensor):
        x = x.to(self.device)
        B, L = x.shape[0], x.shape[1]

        # diffuse
        noise = torch.randn_like(x)
        assert noise_levels.shape == (B,) # or noise_level.shape == (B, L)
        alphas_sqrt = self.alphas[noise_levels].sqrt()
        one_minus_alphas_sqrt = (1 - self.alphas[noise_levels]).sqrt()
        while alphas_sqrt.ndim < x.ndim:
            alphas_sqrt = alphas_sqrt.unsqueeze(-1)
            one_minus_alphas_sqrt = one_minus_alphas_sqrt.unsqueeze(-1)
            assert alphas_sqrt.ndim == one_minus_alphas_sqrt.ndim
        
        x_noise = alphas_sqrt * x + one_minus_alphas_sqrt * noise
        return x_noise, noise
    
    def denoise(self, xt: torch.tensor, noise_pred: torch.tensor, t: torch.tensor):
        '''t is the xt's noise levels in [0, T). t can be two dimension tensor.'''
        assert xt.shape == noise_pred.shape
        xt, noise_pred, t = xt.to(self.device), noise_pred.to(self.device), t.to(self.device)

        beta_t, one_minus_beta, alpha_t = self.betas[t], self.one_minus_betas[t], self.alphas[t]
        while beta_t.ndim < xt.ndim:
            beta_t = beta_t.unsqueeze(-1)
            one_minus_beta = one_minus_beta.unsqueeze(-1)
            alpha_t = alpha_t.unsqueeze(-1)
            assert beta_t.ndim == one_minus_beta.ndim and beta_t.ndim == alpha_t.ndim
        
        noise = torch.randn_like(xt)
        mask = (t == 0)
        noise[mask] = 0
        
        x_t_1 = (1/(one_minus_beta.sqrt())) * (xt - (beta_t / ((1 - alpha_t).sqrt())) * noise_pred) + beta_t.sqrt() * noise
        return x_t_1
    
    def denoise_ddim(self, xt: torch.tensor, noise_pred: torch.tensor, t: torch.tensor, t_next: torch.tensor, eta=0.0):
        '''t is the xt's noise levels in [0, T).
           t_next is the xt's noise levels in [0, T) after denoise.
           t < 0 to indicate the demoise process has already completed, no need to denoise further
           when t<0 and t=0, for these two cases, alpha_t_next can be arbitary value(negative value recommend). This special case is processed
           in alpha_t[t<0] = 1 and alpha_t_next[t<=0] = 1'''
        assert xt.shape == noise_pred.shape and t.shape == t_next.shape
        assert (t > t_next).sum() == t.numel(), f'got {(t > t_next).sum()} and {t.numel()}' # t must greater than t_next
        xt, noise_pred, t, t_next = xt.to(self.device), noise_pred.to(self.device), t.to(self.device), t_next.to(self.device)

        alpha_t, alpha_t_next = self.alphas[t], self.alphas[t_next]
        alpha_t[t<0] = 1 # process denoise already completed case.
        alpha_t_next[t<=0] = 1 # process the last denoise step and denoise already completed cases.
        while alpha_t.ndim < xt.ndim:
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_t_next = alpha_t_next.unsqueeze(-1)
            assert alpha_t.ndim == alpha_t_next.ndim
        one_minus_alpha_t, one_minus_alpha_t_next = 1 - alpha_t, 1 - alpha_t_next

        x0_pred = (xt - one_minus_alpha_t.sqrt()*noise_pred) / (alpha_t.sqrt())

        sigma = eta*(
            (((1-alpha_t_next)/(1-alpha_t))*(1-(alpha_t)/(alpha_t_next))).sqrt()
        )
        x_t_direction = (one_minus_alpha_t_next-sigma**2).sqrt()*noise_pred
        noise = torch.randn_like(xt)
        mask = (t == 0)
        noise[mask] = 0

        x_t_next = alpha_t_next.sqrt()*x0_pred + x_t_direction + sigma * noise
        return x_t_next