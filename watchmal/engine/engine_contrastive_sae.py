"""
Class for training a fully supervised classifier
"""

# hydra imports
from hydra.utils import instantiate

# torch imports
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from geomloss import SamplesLoss

# generic imports
from math import floor, ceil
import numpy as np
from numpy import savez
import os
from time import strftime, localtime, time
import sys
from sys import stdout
import copy
import matplotlib.pyplot as plt
#import umap.umap_ as umap

# WatChMaL imports
from watchmal.dataset.data_utils import get_data_loader
from watchmal.utils.logging_utils import CSVData
from watchmal.engine.losses.modded_triplets import TripletMarginLossModded

#Clustering imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

class VMBDLSEngine:
    """Engine for performing training or evaluation  for a classification network."""
    def __init__(self, model, rank, gpu, dump_path, label_set=None):
        """
        Parameters
        ==========
        model
            nn.module object that contains the full network that the engine will use in training or evaluation.
        rank : int
            The rank of process among all spawned processes (in multiprocessing mode).
        gpu : int
            The gpu that this process is running on.
        dump_path : string
            The path to store outputs in.
        label_set : sequence
            The set of possible labels to classify (if None, which is the default, then class labels in the data must be
            0 to N)
        """
        # create the directory for saving the log and dump files
        self.epoch = 0.
        self.step = 0
        self.best_validation_loss = 1.0e10
        self.dirpath = dump_path
        self.rank = rank
        self.model = model
        self.device = torch.device(gpu)
        self.class_num = 4
        self.lower_bound = 0.05
        
        # Setup the parameters to save given the model type
        if isinstance(self.model, DDP):
            self.is_distributed = True
            self.model_accs = self.model.module
            self.ngpus = torch.distributed.get_world_size()
        else:
            self.is_distributed = False
            self.model_accs = self.model

        self.data_loaders = {}
        self.label_set = label_set

        # define the placeholder attributes
        self.data = None
        self.labels = None
        self.vars = None
        self.loss = None

        # logging attributes
        self.train_log = CSVData(self.dirpath + "log_train_{}.csv".format(self.rank))

        if self.rank == 0:
            self.val_log = CSVData(self.dirpath + "log_val.csv")

        self.eval_log = CSVData(self.dirpath + "log_eval.csv")
        self.optimizer = None
        self.scheduler = None
        
        #self.device = "cuda"
        #self.model = model.to(self.device)
        
        #self.miner = TripletMarginMiner(margin=0.2, type_of_triplets="all")
        self.loss_func = SamplesLoss("sinkhorn", blur=0.05,scaling = 0.95,diameter=0.01,debias=True)
       
        ## Earlier resnet layers
        self.layer_size = self.model_accs.enc_out_dim + self.model_accs.latent_dim
       
        if self.model_accs.enc_type == 'resnet18':
            self.layer_size += 448
        elif self.model_accs.enc_type == 'wresnet':
            self.layer_size += 1120
        elif self.model_accs.enc_type == "resnet50":
            self.layer_size += 1792
        else:
            self.layer_size += 300

        self.model_accs.use_sinkhorn = True

    def configure_optimizers(self, optimizer_config):
        """Instantiate an optimizer from a hydra config."""
        self.optimizer = instantiate(optimizer_config, params=self.model_accs.parameters())

    def configure_scheduler(self, scheduler_config):
        """Instantiate a scheduler from a hydra config."""
        self.scheduler = instantiate(scheduler_config, optimizer=self.optimizer)
        print('Successfully set up Scheduler')


    def configure_data_loaders(self, data_config, loaders_config, is_distributed, seed):
        """
        Set up data loaders from loaders hydra configs for the data config, and a list of data loader configs.

        Parameters
        ==========
        data_config
            Hydra config specifying dataset.
        loaders_config
            Hydra config specifying a list of dataloaders.
        is_distributed : bool
            Whether running in multiprocessing mode.
        seed : int
            Random seed to use to initialize dataloaders.
        """
        for name, loader_config in loaders_config.items():
            print("configuring data loaders")
            self.data_loaders[name] = get_data_loader(**data_config, **loader_config, is_distributed=is_distributed, seed=seed)
            if self.label_set is not None:
                self.data_loaders[name].dataset.map_labels(self.label_set)
        #return data_config, loaders_config, is_distributed, seed
    
    def get_synchronized_metrics(self, metric_dict):
        """
        Gathers metrics from multiple processes using pytorch distributed operations for DistributedDataParallel

        Parameters
        ==========
        metric_dict : dict of torch.Tensor
            Dictionary containing values that are tensor outputs of a single process.
        
        Returns
        =======
        global_metric_dict : dict of torch.Tensor
            Dictionary containing concatenated list of tensor values gathered from all processes
        """
        global_metric_dict = {}
        for name, array in zip(metric_dict.keys(), metric_dict.values()):
            tensor = torch.as_tensor(array).to(self.device)
            global_tensor = [torch.zeros_like(tensor).to(self.device) for i in range(self.ngpus)]
            torch.distributed.all_gather(global_tensor, tensor)
            global_metric_dict[name] = torch.cat(global_tensor)
        
        return global_metric_dict
    def backward(self):
        self.loss.backward()  
       
        self.optimizer.step()
        self.optimizer.zero_grad()

    def train_step(self):
        x = self.data.to(self.device)
        y = self.labels.to(self.device)
        vars = self.vars.to(self.device)
        cur_classes = torch.unique(y).long()
        z, cond_x, rand_z = self.model(x, y,vars, device = self.device)
        
        ng_loss = self.loss_func(torch.cat([z,cond_x],1), torch.cat([rand_z,cond_x],1)) ### noise generator losss, conditional params added to compute also loss for generating noise close to this from other conditionals
        #triplets = self.miner(z,y)
        contrastive_loss = self.metric_loss(z,y)
        self.loss = contrastive_loss + ng_loss
        logs = {
            "loss": self.loss.item(),
            "kl_loss": ng_loss.item(),
            "metric_loss": contrastive_loss.item(),
        }
        return logs


        
#Main evaluation loop
    def test(self, val_iter, num_val_batches):
        test_outputs = []
        self.model.eval()
        with torch.no_grad():
            for val_batch in range(num_val_batches):
                try:
                    val_data = next(val_iter)
                except StopIteration:
                    del val_iter
                    print("Fetching new validation iterator...")
                    val_iter = iter(self.data_loaders["validation"])
                    val_data = next(val_iter)

                # extract the event data from the input data tuple
                self.data = val_data['data']
                self.labels = val_data['labels']
                out = self.test_step()
                test_outputs.append(list(out))
        print("evaluating")   
        self.test_epoch_end(test_outputs)
        # return model to training mode
        self.model.train()
        
        return None
    
    def test_step(self):
        x = self.data.to(self.device)
        y = self.labels.to(self.device)
        z= self.model_accs.fc_out(self.model_accs.encoder(x))
        
        l_features = [torch.flatten(self.model_accs.encoder.get_layer_output(x,i),1) for i in range(1,5)] + [z]
        l_features = torch.hstack(l_features)
      
        cur_classes = torch.unique(y)     
           
        return z,y,x,l_features
                
    def test_epoch_end(self, outputs):
        all_data = torch.vstack([x[0] for x in outputs]).cpu().numpy()
        all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()
        all_known_data = all_data[all_labels < self.class_num,...]
        all_known_labels = all_labels[all_labels < self.class_num]
        #all_early_features = torch.vstack([x[3] for x in outputs]).cpu().numpy()

        all_known_data = all_data[all_labels < self.class_num,...]
        
        if self.log_tsne:
            #umapT = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
            x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_known_data)
            x_te_proj_df = pd.DataFrame(x_te_proj_pca[:, :2], columns=['Proj1', 'Proj2'])
            
            x_te_proj_df['label'] = all_known_labels
            fig = plt.figure()
            ax = sns.scatterplot(x = 'Proj1', y = 'Proj2', data=x_te_proj_df,
                    palette='tab20',
                    hue='label',
                    linewidth=0,
                    alpha=0.6,
                    s=7)
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.savefig(self.dirpath+ "/tsne_" + str(self.epoch) + ".png")
            self.save_state("BEST")

        return 
    
    def eval_epoch_end(self, outputs):
        # Used to return a dataframe as well as a TSNE plot at the end of an eval epoch
        # The returned dataframe shall contain :
        #   - all_known_data
        #   - label
        #   - 2D and 3D TSNE components, labelled proj2d_i , proj3d_i

        
        all_data = torch.vstack([x[0] for x in outputs]).cpu().numpy()
        all_labels = torch.hstack([x[1] for x in outputs]).cpu().numpy()
        all_known_data = all_data[all_labels < self.class_num,...]
        all_known_labels = all_labels[all_labels < self.class_num]
        #all_early_features = torch.vstack([x[3] for x in outputs]).cpu().numpy()

        #all_data = all_early_features
        all_known_data = all_data[all_labels < self.class_num,...]

        data_coords = ["z_" + str(i) for i in range(all_known_data.shape[1])]

        eval_df = pd.DataFrame(all_known_data, columns = data_coords)
        eval_df["labels"] = all_labels

        #temporary return to bypass tsne for speed issues
        return
    
        #2D TSNE
        x_te_proj_pca = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(all_known_data)
        x_te_proj_df_2d = pd.DataFrame(x_te_proj_pca[:, :2], columns=['proj2d_1', 'proj2d_2'])
        eval_df = eval_df.merge(x_te_proj_df_2d)
        fig = plt.figure()
        ax = sns.scatterplot('proj2d_1', 'proj2d_2', data=eval_df,
                hue='labels',
                linewidth=0,
                alpha=0.6,
                s=7)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirpath+ "/tsne2d_eval.png")
        plt.clf()

        #3D TSNE
        x_te_proj_pca = TSNE(n_components=3, perplexity=30, learning_rate=200).fit_transform(all_known_data)
        x_te_proj_df_3d = pd.DataFrame(x_te_proj_pca[:, :2], columns=['proj3d_1', 'proj3d_2', "proj3d_3"])
        eval_df = eval_df.merge(x_te_proj_df_3d)
        fig = plt.figure()
        ax = sns.scatterplot('proj3d_1', 'proj3d_2', "proj3d_3", data=eval_df,
                hue='labels',
                linewidth=0,
                alpha=0.6,
                s=7)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.dirpath+ "/tsne2d_eval.png") 

        #plt.show()
        #plt.clf()
        return eval_df
    
    

    def train(self, train_config):
        """
        Train the model on the training set.

        Parameters
        ==========
        train_config
            Hydra config specifying training parameters
        """
        # initialize training params
        epochs              = train_config.epochs
        report_interval     = train_config.report_interval
        val_interval        = train_config.val_interval
        num_val_batches     = train_config.num_val_batches
        checkpointing       = train_config.checkpointing
        save_interval = train_config.save_interval if 'save_interval' in train_config else None
       
        self.kl_coeff = train_config.kl_coeff
        
        self.class_num = 4
        self.cov_scaling = train_config.cov_scaling
        self.log_tsne = train_config.log_tsne
        self.is_tested = 0
        #self.gen_weight = gen_weight
        #self.weights = weights
        #self.opt = opt
        #self.ae_features = ae_features
        self.margin_max_distance = train_config.margin_max_distance
        self.sample_count = train_config.sample_count

        self.metric_loss = TripletMarginLossModded(margin=0.1,neg_margin=32)
        #self.recon_weight = train_config.recon_weight
        
        #self.lower_bound = lower_bound
        self.generated_count = 0
        self.seen_test = False
        self.generation_step = train_config.generation_step
        
        # set the iterations at which to dump the events and their metrics
        if self.rank == 0:
            print(f"Training... Validation Interval: {val_interval}")

        # set model to training mode
        self.model.train()
        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer = self.optimizer,gamma = 0.92, step_size = 1)
        print(self.scheduler.get_last_lr())
        
        # initialize epoch and iteration counters
        self.epoch = 0.
        self.iteration = 0
        self.step = 0
        # keep track of the validation loss

        #learning rate warmup
        self.best_validation_loss = 1.0e10
        self.warmup_updates = 0
        self.target_lr = 0.001
        self.warmup_steps = 1000
        self.initial_lr = 0.00001
        self.lr_step = (self.target_lr-self.initial_lr)/self.warmup_steps
        # initialize the iterator over the validation set
        val_iter = iter(self.data_loaders["validation"])
        # global training loop for multiple epochs
        self.test(val_iter, num_val_batches)
        for self.epoch in range(epochs):
            if self.epoch == 0:
                for g in self.optimizer.param_groups:
                    g['lr'] = 0.00001
            if self.rank == 0:
                print('Epoch', self.epoch+1, 'Starting @', strftime("%Y-%m-%d %H:%M:%S", localtime()))
            
            times = []

            start_time = time()
            iteration_time = start_time
            #to do : call getdataloader inside train loop
            train_loader = self.data_loaders["train"]
            self.step = 0
            # update seeding for distributed samplers
            if self.is_distributed:
                train_loader.sampler.set_epoch(self.epoch)
            
            # local training loop for batches in a single epoch 
            for self.step, train_data in enumerate(train_loader):
                
                # Train on batch
                self.data = train_data['data']
                self.labels = train_data['labels']
                self.vars = train_data["cond_vec"]
                # Call forward: make a prediction & measure the average error using data = self.data
                res = self.train_step()
                self.backward()
                # update the epoch and iteration
                # self.epoch += 1. / len(self.data_loaders["train"])
                self.step += 1
                self.iteration += 1
                
                # get relevant attributes of result for logging
                train_metrics = {"iteration": self.iteration, "epoch": self.epoch, "loss": res["loss"]}
                if self.warmup_updates < self.warmup_steps:
                    self.warmup_updates+=1
                    for g in self.optimizer.param_groups:
                        g['lr'] += self.lr_step
                        train_metrics["lr"] = g["lr"]
                elif self.warmup_steps == self.warmup_steps:
                    self.warmup_updates+=1

                    for g in self.optimizer.param_groups:
                        g['lr'] = self.target_lr
                
                for g in self.optimizer.param_groups:
                        train_metrics["lr"] = g["lr"]
                            
                # record the metrics for the mini-batch in the log
                self.train_log.record(train_metrics)
                self.train_log.write()
                self.train_log.flush()
                
                # run validation on given intervals
                if self.iteration % val_interval == 0:
                    self.validate(val_iter, num_val_batches, checkpointing)
                    

                    # if self.epoch == 0:
                    #     for g in self.optimizer.param_groups:
                    #         g['lr'] = 0.01
                    
                # print the metrics at given intervals
                if self.rank == 0 and self.iteration % report_interval == 0:
                    previous_iteration_time = iteration_time
                    iteration_time = time()

                    print("... Iteration %d ... Epoch %d ... Step %d/%d  ... Training Loss %1.3f ... Metric Loss %1.3f ... KL_loss %1.3f ... Learning rate %1.7f ...  Time Elapsed %1.3f ... Iteration Time %1.3f" %
                          (self.iteration, self.epoch+1, self.step, len(train_loader), res["loss"], res["metric_loss"], res["kl_loss"], self.scheduler.get_last_lr()[0], iteration_time - start_time, iteration_time - previous_iteration_time))
             
            
            self.test(val_iter, num_val_batches)
            print('test is done, tsne is done')
            if self.scheduler is not None:
                self.scheduler.step()

            if (save_interval is not None) and ((self.epoch+1)%save_interval == 0):
                self.save_state(name=f'_epoch_{self.epoch+1}')   
      
        self.train_log.close()
        if self.rank == 0:
            self.val_log.close()

    def validate(self, val_iter, num_val_batches, checkpointing):
        """
        Perform validation with the current state, on a number of batches of the validation set.

        Parameters
        ----------
        val_iter : iter
            Iterator of the validation dataset.
        num_val_batches : int
            Number of validation batches to iterate over.
        checkpointing : bool
            Whether to save the current state to disk.
        """
        # set model to eval mode
        self.model.eval()
        val_metrics = {"iteration": self.iteration, "loss": 0., 'metric_loss': 0,
                      'kl_loss' : 0, "saved_best": 0}
        for val_batch in range(num_val_batches):
            try:
                val_data = next(val_iter)
            except StopIteration:
                del val_iter
                print("Fetching new validation iterator...")
                val_iter = iter(self.data_loaders["validation"])
                val_data = next(val_iter)

            # extract the event data from the input data tuple
            self.data = val_data['data']
            self.labels = val_data['labels']
            val_res = self.train_step()

            val_metrics["loss"] += val_res["loss"]
            val_metrics["metric_loss"] += val_res["metric_loss"]
            val_metrics["kl_loss"] += val_res["kl_loss"]
     
        # return model to training mode
        self.model.train()
        # record the validation stats
        val_metrics["loss"] /= num_val_batches
        val_metrics["metric_loss"] /= num_val_batches
        val_metrics["kl_loss"] /= num_val_batches
        local_val_metrics = {"loss": np.array([val_metrics["loss"]]), "metric_loss": np.array([val_metrics["metric_loss"]]), "kl_loss": np.array([val_metrics["kl_loss"]])}

        if self.is_distributed:
            global_val_metrics = self.get_synchronized_metrics(local_val_metrics)
            for name, tensor in zip(global_val_metrics.keys(), global_val_metrics.values()):
                global_val_metrics[name] = np.array(tensor.cpu())
        else:
            global_val_metrics = local_val_metrics

        if self.rank == 0:
            # Save if this is the best model so far
            global_val_loss = np.mean(global_val_metrics["loss"])
            global_val_metricloss = np.mean(global_val_metrics["metric_loss"])
            global_val_klloss = np.mean(global_val_metrics["kl_loss"])

            val_metrics["loss"] = global_val_loss
            val_metrics["metric_loss"] = global_val_metricloss
            val_metrics["kl_loss"] = global_val_klloss
            val_metrics["epoch"] = self.epoch

            if val_metrics["loss"] < self.best_validation_loss:
                self.best_validation_loss = val_metrics["loss"]
                print('best validation loss so far!: {}'.format(self.best_validation_loss))
                self.save_state("BEST")
                val_metrics["saved_best"] = 1

            # Save the latest model if checkpointing
            if checkpointing:
                self.save_state()

            self.val_log.record(val_metrics)
            self.val_log.write()
            self.val_log.flush()

    def evaluate(self, test_config):
        """Evaluate the performance of the trained model on the test set."""
        print("evaluating in directory: ", self.dirpath)

        # Variables to output at the end
        eval_loss = 0.0
        eval_iterations = 0
        
        # Iterate over the validation set to calculate val_loss and val_acc
        with torch.no_grad():
            
            # Set the model to evaluation mode
            self.model.eval()
            
            # Variables for the confusion matrix
            loss, indices, labels, softmaxes= [],[],[],[]
            eval_outputs = []
            # Extract the event data and label from the DataLoader iterator
            for it, eval_data in enumerate(self.data_loaders["test"]):
                
                # load data
                self.data = eval_data['data']
                self.labels = eval_data['labels']

                eval_indices = eval_data['indices']
                
                # Run the forward procedure and output the result
                out = self.test_step()
                out_bis = []
                for t in out:
                    out_bis.append(t.detach().cpu())
                
                for j in range(len(self.labels)):
                    eval_metrics = {"z_"+str(i) : out_bis[0][j][i] for i in range(self.model.latent_dim)}
                    # eval_loss += result['loss']
                    # evalmetric_loss += result['metric_loss']
                    # evalkl_loss += result['kl_loss']
                    
                    # Add the local result to the final result
                    # indices.extend(eval_indices.numpy())
                    # labels.extend(self.labels.numpy())
                    eval_metrics["indices"] = eval_indices[j].numpy()
                    eval_metrics["labels"] = self.labels[j].numpy()
                    self.eval_log.record(eval_metrics)
                    self.eval_log.write()
                    self.eval_log.flush()
                eval_iterations += 1
                if eval_iterations%1000 == 0:
                    print(eval_iterations/len(self.data_loaders["test"]))
                if eval_iterations >= 10000:
                    break

            eval_df = self.eval_epoch_end(eval_outputs)  
            eval_df = eval_df.insert('indices', indices)
        
        # convert arrays to torch tensors
        #print("loss : " + str(eval_sloss/eval_iterations))

        iterations = np.array([eval_iterations])
        loss = np.array([eval_loss])
     

        local_eval_metrics_dict = {"eval_iterations":iterations, "eval_loss":loss}
        
        indices     = np.array(indices)
        labels      = np.array(labels)
        
        local_eval_results_dict = {"indices":indices, "labels":labels}

        if self.is_distributed:
            # Gather results from all processes
            global_eval_metrics_dict = self.get_synchronized_metrics(local_eval_metrics_dict)
            global_eval_results_dict = self.get_synchronized_metrics(local_eval_results_dict)
            
            if self.rank == 0:
                for name, tensor in zip(global_eval_metrics_dict.keys(), global_eval_metrics_dict.values()):
                    local_eval_metrics_dict[name] = np.array(tensor.cpu())
                
                indices     = np.array(global_eval_results_dict["indices"].cpu())
                labels      = np.array(global_eval_results_dict["labels"].cpu())
                
                
        if self.rank == 0:
#            print("Sorting Outputs...")
#            sorted_indices = np.argsort(indices)

            # Save overall evaluation results
            print("Saving Data...")
            np.save(self.dirpath + "indices.npy", indices)#sorted_indices)
            np.save(self.dirpath + "labels.npy", labels)#[sorted_indices])
            # Compute overall evaluation metrics
            val_iterations = np.sum(local_eval_metrics_dict["eval_iterations"])
            val_loss = np.sum(local_eval_metrics_dict["eval_loss"])
           

            print("\nAvg eval loss : " + str(val_loss/val_iterations))
        
    # ========================================================================
    # Saving and loading models

    def save_state(self, name=""):
        """
        Save model weights and other training state information to a file.
        
        Parameters
        ==========
        name
            Suffix for the filename. Should be "BEST" for saving the best validation state.
        
        Returns
        =======
        filename : string
            Filename where the saved state is saved.
        """
        filename = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     name,
                                     ".pth")
        
        # Save model state dict in appropriate from depending on number of gpus
        model_dict = self.model_accs.state_dict()
        
        # Save parameters
        # 0+1) iteration counter + optimizer state => in case we want to "continue training" later
        # 2) network weight
        torch.save({
            'global_step': self.iteration,
            'optimizer': self.optimizer.state_dict(),
            'state_dict': model_dict
        }, filename)
        print('Saved checkpoint as:', filename)
        return filename

    def restore_best_state(self, placeholder):
        """Restore model using best model found in current directory."""
        
        best_validation_path = "{}{}{}{}".format(self.dirpath,
                                     str(self.model._get_name()),
                                     "BEST",
                                     ".pth")
        
        #best_validation_path = "/home/amisery/WatChMaL/outputs/2023-04-21/09-01-10/outputs/AutoencoderBEST.pth"

        self.restore_state_from_file(best_validation_path)
    
    def restore_state(self, restore_config):
        """Restore model and training state from a file given in the `weight_file` entry of the config."""
        self.restore_state_from_file(restore_config.weight_file)

    def restore_state_from_file(self, weight_file):
        """Restore model and training state from a given filename."""
        # Open a file in read-binary mode
        with open(weight_file, 'rb') as f:
            print('Restoring state from', weight_file)

            # torch interprets the file, then we can access using string keys
            checkpoint = torch.load(f)
            
            # load network weights
            self.model_accs.load_state_dict(checkpoint['state_dict'])
            
            # if optim is provided, load the state of the optim
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            
            # load iteration count
            self.iteration = checkpoint['global_step']
        
        print('Restoration complete.')
