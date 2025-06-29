import torch
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

from einops import rearrange
import numpy as np
from utils.utils import DDPM
from tqdm import tqdm
from utils.utils import viz_trajs, resample_trajectory

class Trainer:
    def __init__(
        self,
        train_data_loader: DataLoader,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        mark: str,
        data_path: str,
        city: str,
    ) -> None:
        self.mark, self.data_path, self.city = mark, data_path, city
        if city == 'chengdu':
            # chengdu: [104.03968953679004, 30.655400079856072, 104.12705400673643, 30.730172829483855]
            self.min_lat, self.max_lat, self.min_lon, self.max_lon = 30.655400079856072, 30.73018, 104.03968953679004, 104.12706
        elif city == 'xian':
            # xian: [108.90659955383317, 34.20694207396501, 108.99373658803785, 34.28184317582219]
            self.min_lat, self.max_lat, self.min_lon, self.max_lon = 34.20694207396501, 34.28185, 108.90659955383317, 108.99374
        else:
            raise ValueError('city not in [chengdu, xian]')
        
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ['RANK'])
        self.device = f'cuda:{self.rank}'

        self.model = model
        self.model = self.model.to(self.device)
        self.model = DDP(self.model)
        # compile should be after DDP, refer to https://pytorch.org/docs/main/notes/ddp.html
        self.model = torch.compile(self.model)
        
        self.model.train()

        self.train_data_loader = train_data_loader
        self.optimizer = optimizer
        self.loss_fn = torch.nn.MSELoss()

        self.ddpm = DDPM(1000, True, self.device)

    def train(self):
        step = -1
        for epoch in range(2000000000):
            self.train_data_loader.sampler.set_epoch(epoch)
            for i, (trajs, labels) in tqdm(enumerate(self.train_data_loader)):
                self.model.train()
                trajs, labels = trajs.to(self.device), labels.to(self.device)
                trajs = rearrange(trajs, 'B L C -> B C L').contiguous()

                trajs_noise, noise, noise_levels = self.ddpm.forward(trajs)

                self.optimizer.zero_grad()
                avg_loss = torch.zeros((1,), device=self.device)
                avg_grad_norm = torch.zeros((1,), device=self.device)
                noise_pred = self.model(trajs_noise, noise_levels, labels, 0.1)
                loss = self.loss_fn(noise_pred, noise)
                avg_loss[0] = loss.item()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                avg_grad_norm[0] = grad_norm.item()
                self.optimizer.step()
                step += 1
                # collect training info
                dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
                dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.AVG)

                if step % 10000 == 0:
                    self.sample()

                if step % 10000 == 0 and self.rank == 0:
                    torch.save(self.model.state_dict(), f"/home/zzhang18/proj/Traj_Gen/saved_models/{self.mark}_{step}.pt")

    @torch.no_grad
    def sample(self):
        self.model.eval()

        # prepare test data
        N = 5000
        sample_num_per_rank = int(N/self.world_size)
        assert N % self.world_size == 0
        # load test data
        label_test = np.load(self.data_path + f'/{self.city}/{self.city}_test_label.npy')[0:N]
        lengths = label_test[:,3].astype(np.int32)
        label_test = label_test[sample_num_per_rank * self.rank: sample_num_per_rank * (self.rank + 1)]
        label_mean = np.load(self.data_path + f'/{self.city}/{self.city}_label_mean.npy')
        label_std = np.load(self.data_path + f'/{self.city}/{self.city}_label_std.npy')
        label_test[:, 0] = np.floor((label_test[:, 0] % 86400) / 300)
        label_test[:,1:6] = (label_test[:,1:6] - label_mean) / label_std

        noise = torch.randn(sample_num_per_rank, 2, 200).to(self.device)
        cond = torch.tensor(label_test).to(self.device)
        assert cond.shape[0] == sample_num_per_rank
        ddim_step = np.array(range(0, 1000, 5))
        for noise_idx in reversed(range(len(ddim_step))):
            t = torch.full((sample_num_per_rank,), ddim_step[noise_idx], dtype=torch.long).to(self.device)
            t_next = torch.full((sample_num_per_rank,), ddim_step[noise_idx - 1] if noise_idx > 0 else -1, dtype=torch.long).to(self.device)
            cond_noise_pred = self.model(noise, t, cond, 0)
            noise_pred = cond_noise_pred
            noise = self.ddpm.denoise_ddim(noise, noise_pred, t, t_next, 0)

        # gather all samples
        sample_array = [torch.zeros_like(noise, device=self.device) for _ in range(self.world_size)]
        dist.all_gather(sample_array, noise)

        if self.rank == 0:
            # convert to numpy
            sample_np = torch.cat(sample_array, dim=0).permute(0,2,1).contiguous().cpu().numpy()
            spatio_mean, spatio_std = np.load(self.data_path + f'/{self.city}/{self.city}_traj_mean.npy'), np.load(self.data_path + f'/{self.city}/{self.city}_traj_std.npy')
            sample_np = sample_np.astype(np.float64)
            sample_np = sample_np * spatio_std + spatio_mean

            image = viz_trajs(sample_np, lengths, [self.min_lon, self.max_lon], [self.min_lat, self.max_lat])
            image.save(f"/home/zzhang18/proj/Traj_Gen/saved_images/{self.mark}_{self.rank}_{self.sample_idx}.png")
