"""
Class implementing a mPMT dataset for CNNs in h5 format
"""

# torch imports
from torch import from_numpy
from torch import flip
from torchvision import transforms as tvtf
import torch.nn.functional as F
# generic imports
import numpy as np
import torch
import random 

# WatChMaL imports
from watchmal.dataset.h5_dataset import H5Dataset
import watchmal.dataset.data_utils as du

barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
pmts_per_mpmt = 19


class CNNmPMTDataset(H5Dataset):
    """
    This class loads PMT hit data from an HDF5 file and provides events formatted for CNNs, where the 3D data tensor's
    first dimension is over the channels, corresponding to hit charge of the 19 PMTs within each mPMT, and the second
    and third dimensions are the height and width of the CNN image. Each pixel of the image corresponds to one mPMT,
    with mPMTs arrange in an event-display-like format.
    """

    def __init__(self, h5file, mpmt_positions_file, padding_type=None, transforms=None, mode=['charge'], log_scale = False, normalize = False, collapse_arrays=False, systematic_transform = False):
        """
        Constructs a dataset for CNN data. Event hit data is read in from the HDF5 file and the PMT charge data is
        formatted into an event-display-like image for input to a CNN. Each pixel of the image corresponds to one mPMT
        module, with channels corresponding to each PMT within the mPMT. The mPMTs are placed in the image according to
        a mapping provided by the numpy array in the `mpmt_positions_file`.

        Parameters
        ----------
        h5file: string
            Location of the HDF5 file containing the event data
        mpmt_positions_file: string
            Location of a npz file containing the mapping from mPMT IDs to CNN image pixel locations
        transforms: sequence of string
            List of random transforms to apply to data before passing to CNN for data augmentation. Each element of the
            list should be the name of a method of this class that performs the transformation
        collapse_arrays: bool
            Whether to collapse the image-like CNN arrays to a single channels containing the sum of other channels.
            i.e. provide the sum of PMT charges in each mPMT instead of providing all PMT charges.
        """
        super().__init__(h5file)
        self.h5file = h5file
        self.mpmt_positions = np.load(mpmt_positions_file)['mpmt_image_positions']
        self.data_size = np.max(self.mpmt_positions, axis=0) + 1
        self.norm_transform = tvtf.Normalize(mean=[0.5], std=[0.5])
        self.barrel_rows = [row for row in range(self.data_size[0]) if
                            np.count_nonzero(self.mpmt_positions[:, 0] == row) == self.data_size[1]]
        n_channels = pmts_per_mpmt
        self.data_size = np.insert(self.data_size, 0, n_channels)
        self.mode = mode 

        self.log_scale = log_scale
        self.normalize = normalize

        self.collapse_arrays = collapse_arrays
        self.transforms = du.get_transformations(self, transforms)
        self.systematic_transform = systematic_transform
        self.global_max = 3212.684
        self.log_global_max = np.log(1+self.global_max)

        if padding_type is not None:
            self.padding_type = getattr(self, padding_type)
        else:
            self.padding_type = None

        self.horizontal_flip_mpmt_map = [0, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 17, 16, 15, 14, 13, 18]
        self.vertical_flip_mpmt_map = [6, 5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 15, 14, 13, 12, 17, 16, 18]
        self.coords_list = np.array([[2,4], [3,4], [4,3], [4,2], [4,1], [3,0], [2,0], [1,0], [0,1], [0,2], [0,3], [1,4], [2,3], [3,3], [3,1], [2,1], [1,1], [1,3], [2,2]])

         ################
        
        self.mu_q = 2.634658   #np.mean(train_events)
        self.mu_t = 1115.6687   #np.mean(train_events)
        self.std_q = 6.9462004  #np.std(train_events)
        self.std_t = 263.4307  #np.std(train_events)
        
        ################

    def process_data(self, hit_pmts, hit_data):
        """
        Returns event data from dataset associated with a specific index

        Parameters
        ----------
        hit_pmts: array_like of int
            Array of hit PMT IDs
        hit_data: array_like of float
            Array of PMT hit charges, or other per-PMT data

        Returns
        -------
        data: ndarray
            Array in image-like format (channels, rows, columns) for input to CNN network.
        """
        hit_mpmts = hit_pmts // pmts_per_mpmt
        hit_pmt_in_modules = hit_pmts % pmts_per_mpmt

        hit_rows = self.mpmt_positions[hit_mpmts, 0]
        hit_cols = self.mpmt_positions[hit_mpmts, 1]

        data = np.zeros(self.data_size, dtype=np.float32)
        data[hit_pmt_in_modules, hit_rows, hit_cols] = hit_data

        # fix barrel array indexing to match endcaps in xyz ordering
        barrel_data = data[:, self.barrel_rows, :]
        data[:, self.barrel_rows, :] = barrel_data[barrel_map_array_idxs, :, :]
        #print(data.shape)
        #data = data/np.max(data)
        #data = self.scaler.fit_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
        # collapse arrays if desired
        if self.collapse_arrays:
            data = np.expand_dims(np.sum(data, 0), 0)
        
        return data

    def __getitem__(self, item):
        data_dict = super().__getitem__(item)
        
        #processed_data = from_numpy(self.process_data(self.event_hit_pmts, self.event_hit_charges))
        # if self.systematic_transform:
        #     processed_data = du.apply_transformations(self.transforms, processed_data)
        # else : 
        #     #processed_data = self.norm_event(processed_data)
        #     processed_data = du.apply_random_transformations(self.transforms, processed_data)
        # if self.padding_type is not None:
        #     processed_data = self.padding_type(processed_data)
            
        #processed_data = self.reshape_img(processed_data)

        # select random choices for transformations for charge and time
        rand_choices = []
        if self.transforms is not None:
            rand_choices = [bool(random.getrandbits(1)) for i in range(len(self.transforms))]
        
        if 'charge' in self.mode:
            hit_data = self.event_hit_charges
            if self.normalize:
                hit_data = self.feature_scaling_std(hit_data, self.mu_q, self.std_q)
            
            charge_image = from_numpy(self.process_data(self.event_hit_pmts, hit_data))
            if self.transforms is not None:
                charge_image = du.apply_random_transformations(self.transforms, charge_image, choices = rand_choices)
            if self.padding_type is not None:
                charge_image = self.padding_type(charge_image)
            if self.systematic_transform : 
                charge_image = self.pad(charge_image)
            # if 'charge' in self.collapse_mode:
            #     mean_channel = torch.mean(charge_image, 0, keepdim=True)
            #     std_channel = torch.std(charge_image, 0, keepdim=True)
            #     charge_image = torch.cat((mean_channel, std_channel), 0)
        
        if 'time' in self.mode:
            hit_data = self.event_hit_times
            if self.normalize:
                hit_data = self.feature_scaling_std(hit_data, self.mu_t, self.std_t)

            time_image = from_numpy(self.process_data(self.event_hit_pmts, hit_data))
            time_image = du.apply_random_transformations(self.transforms, time_image, choices = rand_choices)
            if self.padding_type is not None :
                time_image = self.padding_type(time_image)

            # if 'time' in self.collapse_mode:
            #     mean_channel = torch.mean(time_image, 0, keepdim=True)
            #     std_channel = torch.std(time_image, 0, keepdim=True)
            #     time_image = torch.cat((mean_channel, std_channel), 0)

        
        # Merge all channels
        if ('time' in self.mode) and ('charge' in self.mode):
            processed_image = torch.cat((charge_image, time_image), 0)
        elif 'charge' in self.mode:
            processed_image = charge_image
        else:
            processed_image = time_image

        if self.log_scale :
            processed_image = self.log_transform(processed_image)
            
        data_dict["cond_vec"] = torch.tensor(np.concatenate((data_dict["energies"], data_dict["angles"], np.squeeze(data_dict["positions"]))))
        del data_dict["energies"]
        del data_dict["positions"]
        del data_dict["angles"]
        #processed_image = self.log_scaling(processed_image)
        data_dict["data"] = processed_image
        
        return data_dict

    def reshape5x5(self, m_vals):
        out = torch.zeros(5,5)
        for k in range(len(m_vals)):
            out[self.coords_list[k][0],self.coords_list[k][1]] = m_vals[k]
        return out[None,:]

    def reshape_img(self,data):
        t_out = torch.empty(data.shape[-2]*5,data.shape[-1]*5)
        for i in range(data.shape[-2]):
            for j in range(data.shape[-1]):
                mpmt_vals = data[:,i,j]
                #t_out[i:i+5,j:j+5] = 
                t_out[5*i:5*(i+1),5*j:5*(j+1)] = self.reshape5x5(mpmt_vals)
        return t_out[None,:]
    
    def log_scaling(self,data):
        return self.log_transform(data)/self.log_global_max
    def pad(self, data):
        pad_val = (0,0,2,1)
        return F.pad(data,pad_val,"constant",0)
    def norm_global(self,data):
        #divide by global max over the dataset
        return data/self.global_max
    def norm_event(self,data):
        #divide by data's maximum value
        return data/(torch.max(data).item())
    def log_transform(self,data):
        return torch.log(1+data)
    def unlog(self, data):
        return torch.exp(data)-1   
    def unpad(self,data):
        unpad_val = (0,0,-2,-1)
        return F.pad(data,unpad_val,"constant",0)
    
    def horizontal_flip(self, data):
        """
        Takes image-like data and returns the data after applying a horizontal flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        return flip(data[self.horizontal_flip_mpmt_map, :, :], [2])

    def vertical_flip(self, data):
        """
        Takes image-like data and returns the data after applying a vertical flip to the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        return flip(data[self.vertical_flip_mpmt_map, :, :], [1])

    def flip_180(self, data):
        """
        Takes image-like data and returns the data after applying both a horizontal flip to the image. This is
        equivalent to a 180-degree rotation of the image.
        The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        return self.horizontal_flip(self.vertical_flip(data))
 
    def front_back_reflection(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal flip of the left and
        right halves of the barrels and vertical flip of the endcaps. This is equivalent to reflecting the detector
        swapping the front and back of the event-display view. The channels of the PMTs within mPMTs also have the
        appropriate permutation applied.
        """

        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2                     # 5
        half_barrel_width = data.shape[2]//2                    # 20
        l_endcap_index = half_barrel_width - radius_endcap      # 15
        r_endcap_index = half_barrel_width + radius_endcap      # 25
        
        transform_data = data.clone()

        # Take out the left and right halves of the barrel
        left_barrel = data[:, self.barrel_rows, :half_barrel_width]
        right_barrel = data[:, self.barrel_rows, half_barrel_width:]
        # Horizontal flip of the left and right halves of barrel
        transform_data[:, self.barrel_rows, :half_barrel_width] = self.horizontal_flip(left_barrel)
        transform_data[:, self.barrel_rows, half_barrel_width:] = self.horizontal_flip(right_barrel)

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical flip of the top and bottom endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.vertical_flip(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.vertical_flip(bottom_endcap)

        return transform_data

    def rotation180(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with horizontal and vertical flip of the
        endcaps and shifting of the barrel rows by half the width. This is equivalent to a 180-degree rotation of the
        detector about its axis. The channels of the PMTs within mPMTs also have the appropriate permutation applied.
        """
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]   # 10,18 respectively
        radius_endcap = barrel_row_start//2                 # 5
        l_endcap_index = data.shape[2]//2 - radius_endcap   # 15
        r_endcap_index = data.shape[2]//2 + radius_endcap   # 25   

        transform_data = data.clone()

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index]
        # Vertical and horizontal flips of the endcaps
        transform_data[:, :barrel_row_start, l_endcap_index:r_endcap_index] = self.flip_180(top_endcap)
        transform_data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index] = self.flip_180(bottom_endcap)

        # Swap the left and right halves of the barrel
        transform_data[:, self.barrel_rows, :] = torch.roll(transform_data[:, self.barrel_rows, :], 20, 2)

        return transform_data
    
    def mpmt_padding(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with part of the barrel duplicated to one
        side, and copies of the end-caps duplicated, rotated 180 degrees and with PMT channels in the mPMTs permuted, to
        provide two 'views' of the detect in one image.
        """
        w = data.shape[2]
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        l_endcap_index = w//2 - 5
        r_endcap_index = w//2 + 4

        padded_data = torch.cat((data, torch.zeros_like(data[:, :, :w//2])), dim=2)
        padded_data[:, self.barrel_rows, w:] = data[:, self.barrel_rows, :w//2]

        # Take out the top and bottom endcaps
        top_endcap = data[:, :barrel_row_start, l_endcap_index:r_endcap_index+1]
        bottom_endcap = data[:, barrel_row_end+1:, l_endcap_index:r_endcap_index+1]

        padded_data[:, :barrel_row_start, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l_endcap_index+w//2:r_endcap_index+w//2+1] = self.flip_180(bottom_endcap)

        return padded_data

    def double_cover(self, data):
        """
        Takes CNN input data in event-display-like format and returns the data with all parts of the detector duplicated
        and rearranged to provide a double-cover of the image, providing two 'views' of the detector from a single image
        with less blank space and physically meaningful cyclic boundary conditions at the edges of the image.

        The transformation looks something like the following, where PMTs on the end caps are numbered and PMTs on the
        barrel are letters:
        ```
                             CBALKJIHGFED
             01                01    32
             23                23    10
        ABCDEFGHIJKL   -->   DEFGHIJKLABC
        MNOPQRSTUVWX         PQRSTUVWXMNO
             45                45    76
             67                67    54
                             ONMXWVUSTRQP
        ```
        """
        w = data.shape[2]                                                                            
        barrel_row_start, barrel_row_end = self.barrel_rows[0], self.barrel_rows[-1]
        radius_endcap = barrel_row_start//2
        half_barrel_width, quarter_barrel_width = w//2, w//4

        # Step - 1 : Roll the tensor so that the first quarter is the last quarter
        padded_data = torch.roll(data, -quarter_barrel_width, 2)

        # Step - 2 : Copy the endcaps and paste 3 quarters from the start, after flipping 180 
        l1_endcap_index = half_barrel_width - radius_endcap - quarter_barrel_width
        r1_endcap_index = l1_endcap_index + 2*radius_endcap
        l2_endcap_index = l1_endcap_index+half_barrel_width
        r2_endcap_index = r1_endcap_index+half_barrel_width

        top_endcap = padded_data[:, :barrel_row_start, l1_endcap_index:r1_endcap_index]
        bottom_endcap = padded_data[:, barrel_row_end+1:, l1_endcap_index:r1_endcap_index]
        
        padded_data[:, :barrel_row_start, l2_endcap_index:r2_endcap_index] = self.flip_180(top_endcap)
        padded_data[:, barrel_row_end+1:, l2_endcap_index:r2_endcap_index] = self.flip_180(bottom_endcap)
        
        # Step - 3 : Rotate the top and bottom half of barrel and concat them to the top and bottom respectively
        barrel_rows_top, barrel_rows_bottom = np.array_split(self.barrel_rows, 2)
        barrel_top_half, barrel_bottom_half = padded_data[:, barrel_rows_top, :], padded_data[:, barrel_rows_bottom, :]
        
        concat_order = (self.flip_180(barrel_top_half), 
                        padded_data,
                        self.flip_180(barrel_bottom_half))

        padded_data = torch.cat(concat_order, dim=1)

        return padded_data

    def feature_scaling_std(self, hit_array, mu, std):
            """
                Scale data using standarization.
            """
            standarized_array = (hit_array - mu)/std
            return standarized_array

    def inv_feature_scaling_std(self, hit_array, mu, std):
        ### Inverse scaling back to original scale
        orig_array = std*hit_array + mu
        return orig_array