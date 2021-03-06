import torch
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
import sys
sys.path.append("../")

from _utils import getLogger

logger = getLogger(__name__)

class ClusterTripletTrainer:
    def __init__(self, train_loader, valid_loader, test_loader, model, cluster_distance, loss_fn, sim_fn, device):

        self.train_loader = train_loader
        self.val_loader = valid_loader 
        self.test_loader = test_loader

        self.model = model
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)

        self.cluster_distance = cluster_distance
        self.loss_fn = loss_fn#ロス関数
        self.sim_fn = sim_fn#類似度関数

        self.device = device
        
        

    def fit(self, lr, n_epochs, log_interval, save_epoch_interval, start_epoch=0, outdir="../result/checkpoint/", data_dirname=None):
        """
        Loaders, model, loss function and metrics should work together for a given task,
        i.e. The model should be able to process data output of loaders,
        loss function should process target output of loaders and outputs from the model

        Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
        Siamese network: Siamese loader, siamese model, contrastive loss
        Online triplet learning: batch loader, embedding model, online triplet loss
        """


        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)

        for epoch in range(0, start_epoch):
            scheduler.step()

        for epoch in range(start_epoch, n_epochs):
            scheduler.step()

            # Train stage
            train_loss = self.train_epoch(optimizer, log_interval)

            message = 'Epoch: {}/{}\n\tTrain set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

            # Validation stage
            val_loss, val_acc_rate = self.validation_epoch()
            val_loss /= len(self.val_loader)

            message += '\n\tValidation set: Average loss: {:.4f}'.format(val_loss)
            message += '\n\t                Accuracy rate: {:.2f}%'.format(val_acc_rate)

            # Test stage
            test_loss, test_acc_rate = self.test_epoch()

            message += '\n\tTest set: Average loss: {:.4f}'.format(test_loss)
            message += '\n\t          Accuracy rate: {:.2f}%'.format(test_acc_rate)

            logging.info(message + "\n")


            if data_dirname is not None and (epoch+1) % save_epoch_interval == 0:
                if torch.cuda.device_count() > 1:
                    num_out = self.model.module.embedding_net.num_out
                    torch.save(self.model.module.embedding_net.state_dict(), f"{outdir}{data_dirname}_embeddingNet_out{num_out}_epoch{epoch}.pth")
                    torch.save(self.model.module.state_dict(), f"{outdir}{data_dirname}_model_out{num_out}_epoch{epoch}.pth")
                else:
                    num_out = self.model.embedding_net.num_out
                    torch.save(self.model.embedding_net.state_dict(), f"{outdir}{data_dirname}_embeddingNet_out{num_out}_epoch{epoch}.pth")
                    torch.save(self.model.state_dict(), f"{outdir}{data_dirname}_model_out{num_out}_epoch{epoch}.pth")

        train_loss = train_loss if float(train_loss) != 0.0 else 10000.0

        return train_loss


    def train_epoch(self, optimizer, log_interval):
        
        self.model.train()
        losses = []
        total_loss = 0
        
        for batch_idx, (data, target, cluster_data) in enumerate(self.train_loader):
            if not type(data) in (tuple, list):
                data = (data,)
            data = tuple(d.to(self.device) for d in data)
            optimizer.zero_grad()
            outputs = self.model(*data)
            anchor_cluster, negative_clusters = cluster_data
            # for i in range(len(anchor_cluster),2):
            anchor_cluster = tuple(d.to(self.device) for d in anchor_cluster)
            anchor_outputs = self.model(*anchor_cluster)
            anchor_outputs = torch.stack(anchor_outputs)
            anchor_outputs = anchor_outputs.reshape([anchor_outputs.shape[1],anchor_outputs.shape[0],anchor_outputs.shape[2]])
            dists = list()
            for negative_cluster in negative_clusters:
                negative_cluster = tuple(d.to(self.device) for d in negative_cluster)
                negative_outputs = self.model(*negative_cluster)
                negative_outputs = torch.stack(negative_outputs)
                negative_outputs = negative_outputs.reshape([negative_outputs.shape[1],negative_outputs.shape[0],negative_outputs.shape[2]])
                dist = torch.tensor([self.cluster_distance(anchor_outputs[i], negative_outputs[i]) for i in range(anchor_outputs.shape[0])], dtype=torch.float64)
                dists.append(dist.unsqueeze(0))
            dists = torch.cat(dists)
            
            min_dist_indexs = dists.argmin(0)
            tuples0, tuples1, tuples2 = list(), list(), list()
            for batch, index in enumerate(min_dist_indexs):
                tuple0, tuple1, tuple2 = negative_clusters[index][0][batch],negative_clusters[index][1][batch],negative_clusters[index][2][batch]
                tuples0.append(tuple0.unsqueeze(0))
                tuples1.append(tuple1.unsqueeze(0))
                tuples2.append(tuple2.unsqueeze(0))
            min_negative_cluster = torch.cat(tuples0), torch.cat(tuples1), torch.cat(tuples2)

            min_negative_cluster = tuple(d.to(self.device) for d in min_negative_cluster)
            min_negative_outputs = self.model(*min_negative_cluster)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            # if type(anchor_outputs) not in (tuple, list):
            #     anchor_outputs = (anchor_outputs,)
            # if type(min_negative_outputs) not in (tuple, list):
            #     min_negative_outputs = (min_negative_outputs,)
            min_negative_outputs = torch.stack(min_negative_outputs)
            min_negative_outputs = min_negative_outputs.reshape([min_negative_outputs.shape[1],min_negative_outputs.shape[0],min_negative_outputs.shape[2]])
            dist = torch.tensor([self.cluster_distance(anchor_outputs[i], min_negative_outputs[i]) for i in range(anchor_outputs.shape[0])], dtype=torch.float64)
            dist = dist.to(self.device)
            loss_outputs = self.loss_fn(*outputs, dist)


            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), np.mean(losses))
                
                logging.info(message)
                losses = []

        total_loss /= (batch_idx + 1)
        
        return total_loss


    def validation_epoch(self):
        with torch.no_grad():
            
            self.model.eval()
            val_loss = 0

            n_true = 0
            for batch_idx, (data, _) in enumerate(self.val_loader):
                
                if not type(data) in (tuple, list):
                    data = (data,)
                
                data = tuple(d.to(self.device) for d in data)
                

                outputs = self.model(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs
                
                loss_outputs = self.loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                val_loss += loss.item()

                pos_dist, neg_dist = self.sim_fn(*loss_inputs)

                for i in range(len(pos_dist)):
                    n_true += 1 if pos_dist[i] < neg_dist[i] else 0

                
            accuracy_rate = (n_true / len(self.val_loader.dataset)) * 100

        return val_loss, accuracy_rate


    def test_epoch(self):
        with torch.no_grad():
            
            self.model.eval()
            
            test_loss = 0
            n_true = 0
            for batch_idx, (data, _) in enumerate(self.test_loader):
                if not type(data) in (tuple, list):
                    data = (data,)
                
                data = tuple(d.to(self.device) for d in data)

                outputs = self.model(*data)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)
                loss_inputs = outputs

                loss_outputs = self.loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                test_loss += loss.item()
                
                pos_dist, neg_dist = self.sim_fn(*loss_inputs)
                
                for i in range(len(pos_dist)):
                    n_true += 1 if pos_dist[i] < neg_dist[i] else 0

            accuracy_rate = (n_true / len(self.test_loader.dataset)) * 100

        return test_loss, accuracy_rate





