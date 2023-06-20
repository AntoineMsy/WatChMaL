"""
Class implementing a dataset for PointNet in h5 format
"""

# generic imports
import numpy as np
from torch import from_numpy
import torch
# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du
from sklearn.preprocessing import StandardScaler

class PointNetDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for point-cloud networks, where the 2D
    data tensor's first dimension is over the channels, using the detector geometry to provide the PMT 3D positions as
    the first three channels, then optionally the PMT orientations, charge and time of the hits as additional channels.
    The second dimension of the data tensor is over the hit PMTs of the event.
    """

    def __init__(self, h5file, geometry_file, use_times=True, use_orientations=False, transforms=None):
        """
        Constructs a dataset for PointNet data. Event hit data is read in from the HDF5 file and the PMT charge and/or
        time data is formatted into an array of points, with x, y and z position and other channels for orientation,
        charge and/or time. Charge is always included but time and orientation channels are optional. The PMT positions
        and orientations are taken from a separate compressed numpy file of the detector geometry.

        Parameters
        ----------
        h5file: string
            Location of the HDF5 file containing the event data
        geometry_file: string
            Location of an npz file containing the position and orientation of PMTs
        use_times: bool
            Whether to use PMT hit times as one of the initial PointNet channels. True by default.
        use_orientations: bool
            Whether to use PMT orientation as some of the initial PointNet channels. False by default.
        n_points: int
            Number of points to pass to the PointNet network. If there are fewer hits in an event than `n_points`, then
            additional points are added filled with zeros. If there are more hits in an event than `n_points`, then the
            hit data is truncated and only the first `n_points` hits are passed to the network.
        transforms
            List of random transforms to apply to data before passing to CNN for data augmentation. Each element of the
            list should be the name of a function in watchmal.dataset.pointnet.transformations that performs the
            transformation.
        """
        super().__init__(h5file)
        geo_file = np.load(geometry_file, 'r')
        self.geo_positions = geo_file["position"].astype(np.float32)
        
        geopos_scaler = StandardScaler()
        self.geo_positions = geopos_scaler.fit_transform(self.geo_positions)
        self.geo_orientations = geo_file["orientation"].astype(np.float32)
        self.use_orientations = use_orientations
        self.use_times = use_times
        
        self.transforms = du.get_transformations(transformations, transforms)
        self.channels = 4
        if use_orientations:
            self.channels += 3
        if use_times:
            self.channels += 1

        ################
        
        self.mu_q = 2.634658   #np.mean(train_events)
        self.mu_t = 1115.6687   #np.mean(train_events)
        self.std_q = 6.9462004  #np.std(train_events)
        self.std_t = 263.4307  #np.std(train_events)
        
        ################

    def __getitem__(self, item):
        # if type(item) == list :
        #     n_pad = item[1]
        #     item = item[0] 
        # else : 
        #     n_pad = 0
        data_dict = super().__getitem__(item)
        n_hits = self.event_hit_pmts.shape[0]
        n_pad = 256 - n_hits%256
        hit_positions = from_numpy(self.geo_positions[self.event_hit_pmts[:n_hits], :])
        data = torch.zeros((self.channels, n_hits + n_pad))
        data[:3, :n_hits] = hit_positions.T
        if self.use_orientations:
            hit_orientations = from_numpy(self.geo_orientations[self.event_hit_pmts[:n_hits], :])
            data[3:6, :n_hits] = hit_orientations.T
        if self.use_times:
            data[-2, :n_hits] = from_numpy(self.feature_scaling_std(self.event_hit_times[:n_hits], self.mu_t, self.std_t))
        data[-1, :n_hits] = from_numpy(self.feature_scaling_std(self.event_hit_charges[:n_hits], self.mu_q, self.std_q))

        data = du.apply_random_transformations(self.transforms, data)

        data_dict["data"] = data
        data_dict["item"] = torch.tensor(item)
        return data_dict
    
    def feature_scaling_std(self, hit_array, mu, std):
        """
            Scale data using standarization.
        """
        standarized_array = (hit_array - mu)/std
        return standarized_array
