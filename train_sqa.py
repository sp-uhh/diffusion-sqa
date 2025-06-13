# Adapted from
# https://github.com/NVlabs/edm2/blob/main/train_edm2.py.
# and
# https://github.com/NVlabs/edm2/blob/main/training/training_loop.py.

"""
Train diffusion models for speech quality assessment for the paper "Non-intrusive Speech 
Quality Assessment with Diffusion Models Trained on Clean Speech", according to the EDM2 recipe 
from the paper "Analyzing and Improving the Training Dynamics of Diffusion Models".
"""

import sys
sys.path.append('./edm2')

import os
import time
import copy
import pickle
import psutil
import json
import click
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from edm2 import dnnlib
from edm2.torch_utils import distributed as dist
from edm2.torch_utils import training_stats
from edm2.torch_utils import persistence
from edm2.torch_utils import misc


#----------------------------------------------------------------------------
# Main training loop.

def training_loop(
    dataset_kwargs      = dict(class_name='src.dataset_audio.AudioFolderDataset', path=None),
    encoder_kwargs      = dict(class_name='src.encoders.MelSpectrogramEncoder'),
    data_loader_kwargs  = dict(class_name='torch.utils.data.DataLoader', pin_memory=True, num_workers=2, prefetch_factor=2),
    network_kwargs      = dict(class_name='src.networks_sqa.Precond'),
    loss_kwargs         = dict(class_name='src.loss.EDM2Loss'),
    optimizer_kwargs    = dict(class_name='torch.optim.Adam', betas=(0.9, 0.99)),
    lr_kwargs           = dict(func_name='edm2.training.training_loop.learning_rate_schedule'),
    ema_kwargs          = dict(class_name='edm2.training.phema.PowerFunctionEMA'),

    run_dir             = '.',      # Output directory.
    seed                = 0,        # Global random seed.
    batch_size          = 128,     # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU. None = no limit.
    total_nwav          = 32<<20,    # Train for a total of N training utterances.
    slice_nwav          = None,     # Train for a maximum of N training utterances in one invocation. None = no limit.
    status_nwav         = 1024<<10,  # Report status every N training utterances. None = disable.
    snapshot_nwav       = 2048<<20,    # Save network snapshot every N training utterances. None = disable.
    checkpoint_nwav     = 4096<<20,  # Save state checkpoint every N training utterances. None = disable.

    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    force_finite        = True,     # Get rid of NaN/Inf gradients before feeding them to the optimizer.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
):
    # Initialize.
    prev_status_time = time.time()
    misc.set_random_seed(seed, dist.get_rank())
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    writer = SummaryWriter(os.path.join(run_dir, "logs"))

    # Validate batch size.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()
    assert total_nwav % batch_size == 0
    assert slice_nwav is None or slice_nwav % batch_size == 0
    assert status_nwav is None or status_nwav % batch_size == 0
    assert snapshot_nwav is None or (snapshot_nwav % batch_size == 0 and snapshot_nwav % 1024 == 0)
    assert checkpoint_nwav is None or (checkpoint_nwav % batch_size == 0 and checkpoint_nwav % 1024 == 0)

    # Setup dataset, encoder, and network.

    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
    ref_image = dataset_obj[0]
    dist.print0(f"Dataset length: {len(dataset_obj)}")

    dist.print0('Setting up encoder...')
    encoder = dnnlib.util.construct_class_by_name(**encoder_kwargs)
    ref_image = encoder.encode(torch.as_tensor(ref_image).to(device).unsqueeze(0))

    dist.print0('Constructing network...')
    interface_kwargs = dict(freq_resolution=ref_image.shape[-2], audio_channels=ref_image.shape[1])
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs)
    net.train().requires_grad_(True).to(device)

    # Print network summary.
    if dist.get_rank() == 0:
        misc.print_module_summary(net, [
            torch.zeros([batch_gpu, net.audio_channels, net.freq_resolution, net.freq_resolution], device=device),
            torch.ones([batch_gpu], device=device)
        ], max_nesting=2)

    # Setup training state.
    dist.print0('Setting up training state...')
    state = dnnlib.EasyDict(cur_nwav=0, total_elapsed_time=0)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device])
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs)
    ema = dnnlib.util.construct_class_by_name(net=net, **ema_kwargs) if ema_kwargs is not None else None

    # Load previous checkpoint and decide how long to train.
    checkpoint = dist.CheckpointIO(state=state, net=net, loss_fn=loss_fn, optimizer=optimizer, ema=ema)
    checkpoint.load_latest(run_dir)
    stop_at_nwav = total_nwav
    if slice_nwav is not None:
        granularity = checkpoint_nwav if checkpoint_nwav is not None else snapshot_nwav if snapshot_nwav is not None else batch_size
        slice_end_nwav = (state.cur_nwav + slice_nwav) // granularity * granularity # round down
        stop_at_nwav = min(stop_at_nwav, slice_end_nwav)
    assert stop_at_nwav > state.cur_nwav
    dist.print0(f'Training from {state.cur_nwav // 1000} kwav to {stop_at_nwav // 1000} kwav:')
    dist.print0()

    # Main training loop.
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed, start_idx=state.cur_nwav)
    dataset_iterator = iter(dnnlib.util.construct_class_by_name(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    prev_status_nwav = state.cur_nwav
    cumulative_training_time = 0
    start_nwav = state.cur_nwav
    stats_jsonl = None

    pbar = tqdm(total=stop_at_nwav, initial=state.cur_nwav, unit='specs', unit_scale=True, dynamic_ncols=True, disable=(dist.get_rank() != 0))
    while True:
        done = (state.cur_nwav >= stop_at_nwav)

        # Report status.
        if status_nwav is not None and (done or state.cur_nwav % status_nwav == 0) and (state.cur_nwav != start_nwav or start_nwav == 0):
            cur_time = time.time()
            state.total_elapsed_time += cur_time - prev_status_time
            cur_process = psutil.Process(os.getpid())
            cpu_memory_usage = sum(p.memory_info().rss for p in [cur_process] + cur_process.children(recursive=True))

            dist.print0(' '.join(['Status:',
                'kwav',         f"{training_stats.report0('Progress/kwav',                              state.cur_nwav / 1e3):<9.1f}",
                'time',         f"{dnnlib.util.format_time(training_stats.report0('Timing/total_sec',   state.total_elapsed_time)):<12s}",
                'sec/tick',     f"{training_stats.report0('Timing/sec_per_tick',                        cur_time - prev_status_time):<8.2f}",
                'sec/kwav',     f"{training_stats.report0('Timing/sec_per_kwav',                        cumulative_training_time / max(state.cur_nwav - prev_status_nwav, 1) * 1e3):<7.3f}",
                'maintenance',  f"{training_stats.report0('Timing/maintenance_sec',                     cur_time - prev_status_time - cumulative_training_time):<7.2f}",
                'cpumem',       f"{training_stats.report0('Resources/cpu_mem_gb',                       cpu_memory_usage / 2**30):<6.2f}",
                'gpumem',       f"{training_stats.report0('Resources/peak_gpu_mem_gb',                  torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}",
                'reserved',     f"{training_stats.report0('Resources/peak_gpu_mem_reserved_gb',         torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}",
            ]))
            cumulative_training_time = 0
            prev_status_nwav = state.cur_nwav
            prev_status_time = cur_time
            torch.cuda.reset_peak_memory_stats()

            # Flush training stats.
            training_stats.default_collector.update()
            if dist.get_rank() == 0:
                if stats_jsonl is None:
                    stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
                fmt = {'Progress/tick': '%.0f', 'Progress/kwav': '%.3f', 'timestamp': '%.3f'}
                items = [(name, value.mean) for name, value in training_stats.default_collector.as_dict().items()] + [('timestamp', time.time())]
                items = [f'"{name}": ' + (fmt.get(name, '%g') % value if np.isfinite(value) else 'NaN') for name, value in items]
                stats_jsonl.write('{' + ', '.join(items) + '}\n')
                stats_jsonl.flush()

            # Update progress and check for abort.
            dist.update_progress(state.cur_nwav // 1000, stop_at_nwav // 1000)
            if state.cur_nwav == stop_at_nwav and state.cur_nwav < total_nwav:
                dist.request_suspend()
            if dist.should_stop() or dist.should_suspend():
                done = True

        # Save network snapshot.
        if snapshot_nwav is not None and state.cur_nwav % snapshot_nwav == 0 and (state.cur_nwav != start_nwav or start_nwav == 0):
            if dist.get_rank() == 0:
                ema_list = ema.get() if ema is not None else optimizer.get_ema(net) if hasattr(optimizer, 'get_ema') else net
                ema_list = ema_list if isinstance(ema_list, list) else [(ema_list, '')]
                for ema_net, ema_suffix in ema_list:
                    data = dnnlib.EasyDict(encoder=encoder, dataset_kwargs=dataset_kwargs, loss_fn=loss_fn)
                    data.ema = copy.deepcopy(ema_net).cpu().eval().requires_grad_(False).to(torch.float16)
                    fname = f'network-snapshot-{state.cur_nwav//1000:07d}{ema_suffix}.pkl'
                    dist.print0(f'Saving {fname} ... ', end='', flush=True)
                    with open(os.path.join(run_dir, fname), 'wb') as f:
                        pickle.dump(data, f)
                    dist.print0('done')
                    del data # conserve memory

        # Save state checkpoint.
        if checkpoint_nwav is not None and (done or state.cur_nwav % checkpoint_nwav == 0) and state.cur_nwav != start_nwav:
            checkpoint.save(os.path.join(run_dir, f'training-state-{state.cur_nwav//1000:07d}.pt'))
            misc.check_ddp_consistency(net)

        # Done?
        if done:
            break

        # Evaluate loss and accumulate gradients.
        batch_start_time = time.time()
        misc.set_random_seed(seed, dist.get_rank(), state.cur_nwav)
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                audio = next(dataset_iterator)
                audio = encoder.encode(audio.to(device))
                loss = loss_fn(net=ddp, data=audio)
                training_stats.report('Loss/loss', loss)
                final_loss = loss.sum().mul(loss_scaling / batch_gpu_total)
                final_loss.backward()
                writer.add_scalar('Loss/train_loss', final_loss, state.cur_nwav)

        # Run optimizer and update weights.
        lr = dnnlib.util.call_func_by_name(cur_nimg=state.cur_nwav, batch_size=batch_size, **lr_kwargs)
        training_stats.report('Loss/learning_rate', lr)
        writer.add_scalar('Loss/LR', lr, state.cur_nwav)

        for g in optimizer.param_groups:
            g['lr'] = lr
        if force_finite:
            for param in net.parameters():
                if param.grad is not None:
                    torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
        optimizer.step()

        # Update EMA and training state.
        state.cur_nwav += batch_size
        if ema is not None:
            ema.update(cur_nimg=state.cur_nwav, batch_size=batch_size)
        cumulative_training_time += time.time() - batch_start_time
        pbar.update(batch_size)
    
    pbar.close()
    writer.close()

#----------------------------------------------------------------------------
# Configuration presets.

config_presets = {
    'edm2-ears16k-xxs': dnnlib.EasyDict(duration=32<<20, batch=128, channels=128, lr=0.0120, decay=70000, dropout=0.00, P_mean=-1.2, P_std=1.2),
}

#----------------------------------------------------------------------------
# Setup arguments for training.training_loop.training_loop().

def setup_training_config(preset='edm2-ears16k-xxs', **opts):
    opts = dnnlib.EasyDict(opts)
    c = dnnlib.EasyDict()
    
    # Preset.
    if preset not in config_presets:
        raise click.ClickException(f'Invalid configuration preset "{preset}"')
    for key, value in config_presets[preset].items():
        if opts.get(key, None) is None:
            opts[key] = value

    # Dataset.
    c.dataset_kwargs = dnnlib.EasyDict(
        class_name='src.dataset_audio.AudioFolderDataset', 
        path=opts.data, 
        segment_size=opts.segment_size
    )
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
        dataset_channels = dataset_obj.raw_shape[1]
        if dataset_channels != 1:
            raise click.ClickException(f'--data: Unsupported channel count {dataset_channels}')
        del dataset_obj # conserve memory
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

    # Encoder.
    c.encoder_kwargs = dnnlib.EasyDict(class_name='src.encoders.MelSpectrogramEncoder')

    # Hyperparameters.
    c.update(total_nwav=opts.duration, batch_size=opts.batch)
    c.network_kwargs = dnnlib.EasyDict(class_name='src.networks_sqa.Precond', model_channels=opts.channels, dropout=opts.dropout)
    c.loss_kwargs = dnnlib.EasyDict(class_name='src.loss.EDM2Loss', P_mean=opts.P_mean, P_std=opts.P_std)
    c.lr_kwargs = dnnlib.EasyDict(func_name='edm2.training.training_loop.learning_rate_schedule', ref_lr=opts.lr, ref_batches=opts.decay)

    # Performance-related options.
    c.batch_gpu = opts.get('batch_gpu', 0) or None
    c.network_kwargs.use_fp16 = opts.get('fp16', True)
    c.loss_scaling = opts.get('ls', 1)
    c.cudnn_benchmark = opts.get('bench', True)

    # I/O-related options.
    c.status_nwav = opts.get('status', 0) or None
    c.snapshot_nwav = opts.get('snapshot', 0) or None
    c.checkpoint_nwav = opts.get('checkpoint', 0) or None
    c.seed = opts.get('seed', 0)
    return c

#----------------------------------------------------------------------------
# Print training configuration.

def print_training_config(run_dir, c):
    dist.print0()
    dist.print0('Training config:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Segment size:            {c.dataset_kwargs.segment_size}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

#----------------------------------------------------------------------------
# Launch training.

def launch_training(run_dir, c):
    if dist.get_rank() == 0 and not os.path.isdir(run_dir):
        dist.print0('Creating output directory...')
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)

    torch.distributed.barrier()
    dnnlib.util.Logger(file_name=os.path.join(run_dir, 'log.txt'), file_mode='a', should_flush=True)
    training_loop(run_dir=run_dir, **c)

#----------------------------------------------------------------------------
# Parse an integer with optional power-of-two suffix:
# 'Ki' = kibi = 2^10
# 'Mi' = mebi = 2^20
# 'Gi' = gibi = 2^30

def parse_nwav(s):
    if isinstance(s, int):
        return s
    if s.endswith('Ki'):
        return int(s[:-2]) << 10
    if s.endswith('Mi'):
        return int(s[:-2]) << 20
    if s.endswith('Gi'):
        return int(s[:-2]) << 30
    return int(s)

#----------------------------------------------------------------------------
# Command line interface.

@click.command()

# Main options.
@click.option('--outdir',           help='Where to save the results', metavar='DIR',            type=str, required=True)
@click.option('--data',             help='Path to the dataset', metavar='ZIP|DIR',              type=str, required=True)
@click.option('--preset',           help='Configuration preset', metavar='STR',                 type=str, default='edm2-ears16k-xxs', show_default=True)

# Hyperparameters.
@click.option('--duration',         help='Training duration', metavar='NWAV',                   type=parse_nwav, default=None)
@click.option('--segment_size',     help='Segment size', metavar='NWAV',                        type=parse_nwav, default=65280) # Corresponds to 256 spectrogram frames
@click.option('--batch',            help='Total batch size', metavar='NWAV',                    type=parse_nwav, default=None)
@click.option('--channels',         help='Channel multiplier', metavar='INT',                   type=click.IntRange(min=64), default=None)
@click.option('--dropout',          help='Dropout probability', metavar='FLOAT',                type=click.FloatRange(min=0, max=1), default=None)
@click.option('--P_mean', 'P_mean', help='Noise level mean', metavar='FLOAT',                   type=float, default=None)
@click.option('--P_std', 'P_std',   help='Noise level standard deviation', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--lr',               help='Learning rate max. (alpha_ref)', metavar='FLOAT',     type=click.FloatRange(min=0, min_open=True), default=None)
@click.option('--decay',            help='Learning rate decay (t_ref)', metavar='BATCHES',      type=click.FloatRange(min=0), default=None)

# Performance-related options.
@click.option('--batch-gpu',        help='Limit batch size per GPU', metavar='NWAV',            type=parse_nwav, default=0, show_default=True)
@click.option('--fp16',             help='Enable mixed-precision training', metavar='BOOL',     type=bool, default=True, show_default=True)
@click.option('--ls',               help='Loss scaling', metavar='FLOAT',                       type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',            help='Enable cuDNN benchmarking', metavar='BOOL',           type=bool, default=True, show_default=True)

# I/O-related options.
@click.option('--status',           help='Interval of status prints', metavar='NWAV',           type=parse_nwav, default='2048Ki', show_default=True)
@click.option('--snapshot',         help='Interval of network snapshots', metavar='NWAV',       type=parse_nwav, default='2048Ki', show_default=True)
@click.option('--checkpoint',       help='Interval of training checkpoints', metavar='NWAV',    type=parse_nwav, default='4096Ki', show_default=True)
@click.option('--seed',             help='Random seed', metavar='INT',                          type=int, default=0, show_default=True)
@click.option('-n', '--dry-run',    help='Print training options and exit',                     is_flag=True)

def cmdline(outdir, dry_run, **opts):
    """Train diffusion models according to the EDM2 recipe from the paper
    "Analyzing and Improving the Training Dynamics of Diffusion Models".

    Examples:

    \b
    # Train model on 4 GPUs
    torchrun --standalone --nproc_per_node=4 train_sqa.py \\
        --outdir=training-runs/edm2-ears16k-xxs \\
        --data=<path-to-dataset> \\
        --batch-gpu=32
    \b
    # To resume training, run the same command again.
    """
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    dist.print0('Setting up training config...')
    c = setup_training_config(**opts)
    print_training_config(run_dir=outdir, c=c)
    if dry_run:
        dist.print0('Dry run; exiting.')
    else:
        launch_training(run_dir=outdir, c=c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
