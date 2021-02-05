import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import numpy as np
import logging
import sys
sys.path.append("../")

from models import TripletNet
from _utils import getLogger

logger = getLogger(__name__)



class UIRTrainer:
    def __init__(self, sup_train_loader, semisup_train_loader, sup_valid_loader, semisup_valid_loader, sup_test_loader, semisup_test_loader, \
                model, margin_penalty, sup_train_loss_fn, semisup_train_loss_fn, test_loss_fn, sim_fn, device):

        self.sup_train_loader = sup_train_loader
        self.sup_val_loader = sup_valid_loader 
        self.sup_test_loader = sup_test_loader
        self.semisup_train_loader = semisup_train_loader
        self.semisup_val_loader = semisup_valid_loader 
        self.semisup_test_loader = semisup_test_loader

        self.model = model
        self.test_model = TripletNet(model)
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
            self.test_model = nn.DataParallel(self.test_model)
        self.margin_penalty = margin_penalty
        self.sup_train_loss_fn = sup_train_loss_fn#ロス関数
        self.semisup_train_loss_fn = semisup_train_loss_fn#ロス関数
        self.test_loss_fn = test_loss_fn#ロス関数
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


        sup_optimizer = optim.Adam([{'params': self.model.parameters()}, {'params': self.margin_penalty.parameters()}], lr=lr)
        sup_scheduler = lr_scheduler.StepLR(sup_optimizer, 8, gamma=0.1, last_epoch=-1)
        semisup_optimizer = optim.Adam([{'params': self.model.parameters()}, {'params': self.margin_penalty.parameters()}], lr=lr)
        semisup_scheduler = lr_scheduler.StepLR(semisup_optimizer, 8, gamma=0.1, last_epoch=-1)

        n_epochs *= 2 #教師あり学習と半教師あり学習の２回行うため
        if start_epoch != 0:
            embedding_model = torch.load(f"{outdir}{data_dirname}_embeddingNet_out{self.model.num_out}_epoch{start_epoch-1}.pth")
            model = torch.load(f"{outdir}{data_dirname}_model_out{self.model.num_out}_epoch{start_epoch-1}.pth")
            margin_penalty = torch.load(f"{outdir}{data_dirname}_marginPenalty_out{self.model.num_out}_epoch{start_epoch-1}.pth")
            self.model.load_state_dict(model)
            self.margin_penalty.load_state_dict(margin_penalty)
            
        for epoch in range(0, start_epoch):
            if epoch < n_epochs/2:
                sup_scheduler.step()
            else:
                semisup_scheduler.step()


        for epoch in range(start_epoch, n_epochs):
            # Train stage
            if epoch < n_epochs/2:
                sup_scheduler.step()
                train_loss = self.sup_train_epoch(sup_optimizer, log_interval)
            else:
                semisup_scheduler.step()
                train_loss = self.semisup_train_epoch(semisup_optimizer, log_interval)

            message = 'Epoch: {}/{}\n\tTrain set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)

            # Validation stage
            sup_val_loss, semisup_val_loss, sup_val_acc, semisup_val_acc = self.validation_epoch()
            sup_val_loss /= len(self.sup_val_loader)
            semisup_val_loss /= len(self.semisup_val_loader)

            message += '\n\tValidation set: Average loss: labeled{:.6f}, unlabeled{:.6f}'.format(sup_val_loss, semisup_val_loss)
            message += '\n\t                Accuracy rate: labeled{:.6f}%, unlabeled{:.6f}%'.format(sup_val_acc, semisup_val_acc)

            # Test stage
            sup_test_loss, semisup_test_loss, sup_test_acc, semisup_test_acc = self.test_epoch()

            message += '\n\tTest set: Average loss: labeled{:.6f}, unlabeled{:.6f}'.format(sup_test_loss, semisup_test_loss)
            message += '\n\t          Accuracy rate: labeled{:.6f}%, unlabeled{:.6f}%'.format(sup_test_acc, semisup_test_acc)

            logging.info(message + "\n")

            if data_dirname is not None and (epoch+1) % save_epoch_interval == 0:
                torch.save(self.model.state_dict(), f"{outdir}{data_dirname}_embeddingNet_out{self.model.num_out}_epoch%d.pth" % epoch)
                torch.save(self.model.state_dict(), f"{outdir}{data_dirname}_model_out{self.model.num_out}_epoch%d.pth" % epoch)
                torch.save(self.margin_penalty.state_dict(),  f"{outdir}{data_dirname}_marginPenalty_out{self.model.num_out}_epoch%d.pth" % epoch)
        train_loss = train_loss if float(train_loss) != 0.0 else 10000.0

        return train_loss


    def sup_train_epoch(self, optimizer, log_interval):
    
        self.model.train()
        losses = []
        total_loss = 0
        

        for batch_idx, (data, target) in enumerate(self.sup_train_loader):
            data = data.to(self.device)
            target = target.to(self.device).long()
        
            optimizer.zero_grad()
            outputs = self.model(data)
            outputs = self.margin_penalty(outputs, target)
            
            loss_outputs = self.sup_train_loss_fn(outputs, target)
            
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.item())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            
            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data), len(self.sup_train_loader.dataset),
                    100. * batch_idx / len(self.sup_train_loader), np.mean(losses))
                
                logging.info(message)
                losses = []

        total_loss /= (batch_idx + 1)
        
        return total_loss

    def semisup_train_epoch(self, optimizer, log_interval):

        self.model.train()
        labeled_losses = []
        unlabeled_losses = []
        losses = []
        total_loss = 0
        
        
        for batch_idx, ((labeled_data, labeled_target), (unlabeled_data, unlabeled_target)) in enumerate(zip(self.sup_train_loader, self.semisup_train_loader)):
            labeled_data, unlabeled_data = labeled_data.to(self.device), unlabeled_data.to(self.device)
            labeled_target, unlabeled_target = labeled_target.to(self.device).long(), unlabeled_target.to(self.device).long()
            
            optimizer.zero_grad()
            
            labeled_outputs, unlabeled_outputs = self.model(labeled_data), self.model(unlabeled_data)
            labeled_outputs, unlabeled_outputs \
                = self.margin_penalty(labeled_outputs, labeled_target), self.margin_penalty(unlabeled_outputs, unlabeled_target)
            
            labeled_loss_outputs, unlabeled_loss_outputs \
                = self.sup_train_loss_fn(labeled_outputs, labeled_target), self.semisup_train_loss_fn(unlabeled_outputs)
            labeled_loss = labeled_loss_outputs[0] if type(labeled_loss_outputs) in (tuple, list) else labeled_loss_outputs
            unlabeled_loss = unlabeled_loss_outputs[0] if type(unlabeled_loss_outputs) in (tuple, list) else unlabeled_loss_outputs
            labeled_losses.append(labeled_loss.item())
            unlabeled_losses.append(unlabeled_loss.item())
            loss = labeled_loss + unlabeled_loss
            total_loss += loss
            loss.backward()
            optimizer.step()

            
            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{}, {}/{} ({:.0f}%)]\tLoss: labeled{:.6f}, unlabeled{:.6f}'.format(
                    batch_idx * len(labeled_data), len(self.sup_train_loader.dataset),  
                    batch_idx * len(unlabeled_data), len(self.semisup_train_loader.dataset), 100. * batch_idx / len(self.sup_train_loader), 
                    np.mean(labeled_losses), np.mean(unlabeled_losses))
                
                logging.info(message)
                losses = []
                labeled_losses = []
                unlabeled_losses = []

        total_loss /= (batch_idx + 1)
        
        return total_loss
        

    def validation_epoch(self):
        with torch.no_grad():
            
            self.test_model.eval()
            
            accuracy_rates = list()
            val_losses = list()
            for val_loader in [self.sup_val_loader, self.semisup_val_loader]:
                val_loss = 0
                n_true = 0
                for batch_idx, (data, _) in enumerate(val_loader):
                    
                    if not type(data) in (tuple, list):
                        data = (data,)
                    
                    data = tuple(d.to(self.device) for d in data)
                    
                    outputs = self.test_model(*data)

                    if type(outputs) not in (tuple, list):
                        outputs = (outputs,)
                    loss_inputs = outputs
                    
                    loss_outputs = self.test_loss_fn(*loss_inputs)
                    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                    val_loss += loss.item()

                    pos_dist, neg_dist = self.sim_fn(*loss_inputs)

                    for i in range(len(pos_dist)):
                        n_true += 1 if pos_dist[i] < neg_dist[i] else 0

                val_losses.append(val_loss)
                accuracy_rates.append((n_true / len(val_loader.dataset)) * 100)
            sup_val_loss, semisup_val_loss = val_losses
            sup_accuracy_rate, semisup_accuracy_rate =  accuracy_rates
        return sup_val_loss, semisup_val_loss, sup_accuracy_rate, semisup_accuracy_rate


    def test_epoch(self):
        
        with torch.no_grad():
            
            self.test_model.eval()
            
            accuracy_rates = list()
            test_losses = list()
            for test_loader in [self.sup_test_loader, self.semisup_test_loader]:
                test_loss = 0
                n_true = 0
                for batch_idx, (data, _) in enumerate(test_loader):
                    
                    if not type(data) in (tuple, list):
                        data = (data,)
                    
                    data = tuple(d.to(self.device) for d in data)
                    

                    outputs = self.test_model(*data)

                    if type(outputs) not in (tuple, list):
                        outputs = (outputs,)
                    loss_inputs = outputs
                    
                    loss_outputs = self.test_loss_fn(*loss_inputs)
                    loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
                    test_loss += loss.item()

                    pos_dist, neg_dist = self.sim_fn(*loss_inputs)

                    for i in range(len(pos_dist)):
                        n_true += 1 if pos_dist[i] < neg_dist[i] else 0

                test_losses.append(test_loss)
                accuracy_rates.append((n_true / len(test_loader.dataset)) * 100)
            sup_test_loss, semisup_test_loss = test_losses
            sup_accuracy_rate, semisup_accuracy_rate =  accuracy_rates
        return sup_test_loss, semisup_test_loss, sup_accuracy_rate, semisup_accuracy_rate





