import torch
from ddp.ddp_utils import ddp_setup, ddp_cleanup
from datasets.traj_dataloader import TrajectoryData
from torch.utils.data import DataLoader
from models.Traj_Transformer import Traj_Transformer
from ddp.ddp_trainer import Trainer
from torch.utils.data.distributed import DistributedSampler

def ddp_main(rank: int, world_size: int, data_path: str, city: str, model_size: str, lon_lat_emb: bool):
    print("ddp setup...")
    ddp_setup(rank, world_size)
    print("ddp setup done.")

    # dataset
    dataset = TrajectoryData(data_path, city)
    dataloader = DataLoader(dataset,
                            batch_size=256,
                            pin_memory=True,
                            shuffle=False,
                            num_workers=4,
                            drop_last=True,
                            prefetch_factor=2,
                            sampler=DistributedSampler(dataset, shuffle=True, seed=0),
    )
    
    # model
    model = Traj_Transformer(
        traj_length=200,
        patch_size=1,
        hidden_size=192,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        lon_lat_embedding=lon_lat_emb
    )
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {total_params}")
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    # train
    mark = f'{model_size}_' + ('2D' if lon_lat_emb else '1D')
    trainer = Trainer(dataloader, model, optimizer, mark, data_path, city)
    print('training start...')
    trainer.train()

    ddp_cleanup()