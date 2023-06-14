"""
Sampler classes
"""

# torch imports
import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

# generic imports
from operator import itemgetter
from typing import Optional


def SubsetSequentialSampler(indices):
    return indices

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = data_source
    def __iter__(self):
        buckets = [[]] * 5000
        yielded = 0
        
        for idx in self.sampler:
            #s = self.sampler.data_source[idx]
            s = self.dataset[idx]
            L = s["data"].shape[1]
            
            # if isinstance(s, tuple):
            #     L = s[0]["mask"].sum()
            # else:
            #     L = s["mask"].sum()
            # if torch.rand(1).item() < 0.1: L = int(1.5*L)
            L = max(0, L // 256)
            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(idx)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        print("loop finished")
        print(leftover)
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                print("end yield")
                print(batch)
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
            
class DistributedBatchSamplerWrapper(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False, batch_size = 10) -> None:
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed,
                         drop_last=drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        indices = list(super().__iter__())
        random_sampler = RandomSampler(data_source=self.dataset)
        batch_sampler = LenMatchBatchSampler(random_sampler, batch_size=self.batch_size, indices=indices)
        return iter(batch_sampler)
    
    def set_epoch(self, epoch):
        """Set the epoch number, used for setting the random seed so that each epoch has a different random seed."""
        self.epoch = epoch
    def __len__(self) -> int:
        return self.num_samples//self.batch_size
              
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper for making general samplers compatible with multiprocessing.

    Allows you to use any sampler in distributed mode when training with 
    torch.nn.parallel.DistributedDataParallel. In such case, each process 
    can pass a DistributedSamplerWrapper instance as a DataLoader sampler, 
    and load a subset of subsampled data of the original dataset that is 
    exclusive to it.
    """

    def __init__(
        self,
        sampler,
        seed,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = False,
    ):
        """
        Initialises a sampler that wraps some other sampler for use with DistributedDataParallel

        Parameters
        ==========
        sampler
            The sampler used for subsampling.
        num_replicas : int, optional
            Number of processes participating in distributed training.
        rank : int, optional
            Rank of the current process within ``num_replicas``.
        shuffle : bool, optional
            If true sampler will shuffle the indices, false by default.
        """
        super(DistributedSamplerWrapper, self).__init__(
            list(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        self.sampler = sampler
        self.epoch = 0
    
    def set_epoch(self, epoch):
        """Set the epoch number, used for setting the random seed so that each epoch has a different random seed."""
        self.epoch = epoch
    
    def __iter__(self):
        # fetch DistributedSampler indices
        indexes_of_indexes = super().__iter__()
        
        # deterministically shuffle based on epoch
        updated_seed = self.seed + int(self.epoch)
        torch.manual_seed(updated_seed)

        # fetch subsampler indices with synchronized seeding
        subsampler_indices = list(self.sampler)

        # get subsampler_indexes[indexes_of_indexes]
        distributed_subsampler_indices = itemgetter(*indexes_of_indexes)(subsampler_indices)

        return iter(distributed_subsampler_indices)
