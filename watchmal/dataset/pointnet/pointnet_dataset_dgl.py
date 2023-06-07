"""
Class implementing a dataset for PointNet in h5 format
"""

# generic imports
import numpy as np
from torch import from_numpy
import torch
import dgl
# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
from watchmal.dataset.pointnet import transformations
import watchmal.dataset.data_utils as du
from sklearn.preprocessing import StandardScaler

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

class PointNetDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for point-cloud networks, where the 2D
    data tensor's first dimension is over the channels, using the detector geometry to provide the PMT 3D positions as
    the first three channels, then optionally the PMT orientations, charge and time of the hits as additional channels.
    The second dimension of the data tensor is over the hit PMTs of the event.
    """

    def __init__(self, h5file, geometry_file, use_times=True, use_orientations=False, n_points=5000, transforms=None):
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
        
        #self.geo_positions = from_numpy(geopos_scaler.fit_transform(self.geo_positions))
        self.geo_positions = from_numpy(self.geo_positions)

        self.relative_positions = self.geo_positions.unsqueeze(0) - self.geo_positions.unsqueeze(1)
        #d = len(self.geo_positions)
        #self.relative_positions = torch.zeros((d,d))
        #self.relative_positions = square_distance(self.geo_positions[None,:], self.geo_positions[None,:]).squeeze(0)

        self.geo_orientations = geo_file["orientation"].astype(np.float32)
        self.use_orientations = use_orientations
        self.use_times = use_times
        self.n_points = n_points
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
        data_dict = super().__getitem__(item)
        n_hits = self.event_hit_pmts.shape[0]
        self.event_hit_pmts = self.event_hit_pmts
        hit_positions = self.geo_positions[self.event_hit_pmts[:n_hits], :]
        
        d_graph = dgl.knn_graph(hit_positions,32)
        d_graph.ndata["pos"] = hit_positions
        #d_graph.ndata["pmt"] = self.event_hit_pmts[:n_hits]
        src, dst = d_graph.edges()
        rel_pos = self.relative_positions[self.event_hit_pmts[dst], self.event_hit_pmts[src], :]
        d_graph.edata["rel_pos"] = rel_pos
        
        #data = np.concatenate((np.full((3, self.n_points), -10e10, dtype=np.float32), np.zeros(self.channels-3, self.n_points)))
        data = np.zeros((self.channels, self.n_points), dtype = np.float32)
        data[:3, :n_hits] = hit_positions.T/100
        if self.use_orientations:
            hit_orientations = self.geo_orientations[self.event_hit_pmts[:n_hits], :]
            data[3:6, :n_hits] = hit_orientations.T
        if self.use_times:
            data[-2, :n_hits] = self.feature_scaling_std(self.event_hit_times[:n_hits], self.mu_t, self.std_q)
        data[-1, :n_hits] = self.feature_scaling_std(self.event_hit_charges[:n_hits], self.mu_q, self.std_q)

        hit_charge = from_numpy(self.feature_scaling_std(self.event_hit_charges[:n_hits], self.mu_q, self.std_q))
        hit_time = from_numpy(self.feature_scaling_std(self.event_hit_times[:n_hits], self.mu_t, self.std_q))
        attr_tensor = torch.stack((hit_charge,hit_time),dim=1)
        data = du.apply_random_transformations(self.transforms, data)
        d_graph.ndata["attr"] = attr_tensor
        #data_dict["data"] = data
        #data_dict["mask"] = np.concatenate((np.ones((3, n_hits), dtype=np.float32), np.zeros((3, self.n_points-n_hits), dtype=np.float32)), axis =1)
        data_dict["nhits"] = n_hits
        data_dict["graph"] = d_graph
        data_dict["event_hits_pmts"] = self.event_hit_pmts
        return data_dict
    
    def feature_scaling_std(self, hit_array, mu, std):
        """
            Scale data using standarization.
        """
        standarized_array = (hit_array - mu)/std
        return standarized_array
