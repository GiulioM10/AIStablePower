# GM 05/17/23
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
from timm.optim.optim_factory import create_optimizer
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
import matplotlib.pyplot as plt

def kaiming_normal(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def init_model(model):
    model.apply(kaiming_normal)
    model.eval()
    return model

class Gym:
    def __init__(self, train_set:DataLoader,
                 val_set: DataLoader,
                 epochs: int, directory:str, 
                 device:torch.device,
                 optimizer: str,
                 loss: str,
                 scheduler: str,
                 lr: float,
                 momentum: float,
                 weight_decay: float,
                 t_initial:int,
                 cycle_mul: float,
                 warmup_t: int
                 ) -> None:
        """This object handles the training proceess and performance assesment of architectures

        Args:
            train_set (DataLoader): DataLoader containing the train spectra
            val_set (DataLoader): DataLoader containing the validation/test spectra
            epochs (int): The numbers of epochs for which we want to train our model
            directory (str): Path to the file in which to store checkpoints
            device (torch.device): Device to be used to perform computations
            learning_rate (float, optional): Learning rate of the optimizer. Defaults to .005.
            weight_decay (float, optional): Weight decay of the optimizer. Defaults to .005.
            momentum (float, optional): Momentum of the optimizer. Defaults to 0.9.
        """
        self.epochs = epochs
        self.directory = directory
        self.device = device
        self.optimizer = optimizer
        self.loss = loss
        self.scheduler = scheduler
        
        self.train_set = train_set
        self.val_set = val_set
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.t_initial = t_initial
        self.cycle_mul = cycle_mul
        self.warmup_t = warmup_t
        
        self.implemented_loss_funcs = {
            "mse": torch.nn.MSELoss,
            "l1": torch.nn.SmoothL1Loss
        }
    
    def _get_optimizer(self, net:torch.nn.Module):
        """Return an optimizer to train a given architecture

        Args:
            net (torch.nn.Module): The architecture to be trained

        Returns:
            torch.optim: An optimizer with the previously specified hyper-parameters
        """
        args = SimpleNamespace()
        args.weight_decay = self.weight_decay
        args.lr = self.lr
        args.opt = self.optimizer
        args.momentum = self.momentum
        
        return create_optimizer(args, net)
    
    def _get_loss(self) -> torch.nn:
        """Returns a loss function to evaluate performance

        Returns:
            torch.nn: A loss function
        """
        if self.loss not in self.implemented_loss_funcs.keys():
            raise Exception(f"Loss {self.loss} is not implemented")
        return self.implemented_loss_funcs[self.loss]()
    
    def _get_lr_scheduler(self, optimizer):
        scheduler = CosineLRScheduler(
            optimizer=optimizer,
            t_initial=self.t_initial,
            lr_min=1e-6,
            cycle_mul=self.cycle_mul,
            warmup_t=self.warmap_t,
            warmup_prefix=True
        )
        return scheduler
    
    def _train(self, net: torch.nn.Module, optimizer, loss_function) -> None:
        """Perform an epoch of training for a network

        Args:
            net (torch.nn.Module): The network to be trained
            optimizer (_type_): An optimizer
            loss_function (_type_): A loss function
        """
        samples = 0
        cumulative_loss = 0.0
        absolute_error = 0.0
        
        net = net.train()
        for spectra, momenta in self.train_set:
            spectra = spectra.to(self.device).half()
            momenta = momenta.to(self.device)
            outputs = net(spectra)

            loss = loss_function(outputs, momenta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            samples += spectra.size(dim=0)
            cumulative_loss += loss.item()
            
            err = torch.abs(outputs - momenta)
            absolute_error += err.sum.item()
        
        return cumulative_loss, absolute_error/samples
        
    def _test(self, net: torch.nn.Module, loss_function):
        """Get performance of a model on a test/validation set

        Args:
            net (torch.nn.Module): The model
            loss_function (_type_): Loss function used to evaluate performance

        Returns:
            tuple: Value of the loss and accuracy
        """
        samples = 0
        cumulative_loss = 0.0
        absolute_error = 0.0
  
        net.eval()

        with torch.no_grad():
            for spectra, momenta in self.val_set:
                spectra = spectra.to(self.device)
                momenta = momenta.to(self.device)
                outputs = net(spectra)

                loss = loss_function(outputs, momenta)

                samples += spectra.size(dim=0)
                cumulative_loss += loss.item()
                
                err = torch.abs(outputs - momenta)
                absolute_error += err.sum.item()

        return cumulative_loss, absolute_error/samples
    
    def workout(self, net: torch.nn.Module, load_checkpoint: bool = False) -> None:
        """Train a model using the equipement of the gym

        Args:
            net (torch.nn.Module): The model to be trained
            load_checkpoint (bool, optional): Wether to load the weights from a previous workout session. Defaults to False.

        Raises:
            Exception: Model already trained for the requested number of epochs
        """
        optimizer = self._get_optimizer(net=net)
        loss_function = self._get_loss()
        scheduler = self._get_lr_scheduler(optimizer)
        
        if load_checkpoint:
            checkpoint = torch.load(self.directory, map_location = self.device)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            epoch = checkpoint['epoch']
            loss_value = checkpoint['loss']
            average_abs_error = checkpoint['average_abs_error']
            results = checkpoint['results']
            if epoch >= self.epochs:
                raise Exception("Model already trained for the desired number of epochs")
            print("------ Epoch {}/{} - Perofrmance on validation set (CHECKPOINT) ------".format(epoch, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, average_abs_error))
            
            
        else:
            net = init_model(net)
            results = {
                "train_loss": [],
                "train_abs_err": [],
                "val_loss": [],
                "val_abs_err": []
            }
            
            
            
            print("------ INITIAL PERFORMANCE ON VALIDATION SET ------")
            loss_value, average_abs_error = self._test(net, loss_function=loss_function)
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, average_abs_error))
            epoch = 0
        
        while epoch < self.epochs:
            train_loss, train_avg_abs_err = self._train(net, optimizer, loss_function)
            loss_value, average_abs_error = self._test(net, loss_function=loss_function)
            print("------ Epoch {}/{} - Perofrmance on train set ------".format(epoch + 1, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(train_loss, train_avg_abs_err))
            print("------ Epoch {}/{} - Perofrmance on validation set ------".format(epoch + 1, self.epochs))
            print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, average_abs_error))
            epoch += 1
            scheduler.step()
            results["val_loss"].append(loss_value)
            results["val_abs_err"].append(average_abs_error)
            results["train_loss"].append(train_loss)
            results["train_abs_err"].append(train_avg_abs_err)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss_value,
                'average_abs_error': average_abs_error,
                'results': results
            }, self.directory)
            
            self.show_learning_curves(True)
        
        print("\n------ PERFORMANCE ON TEST SET AFTER TRAINING ------")
        loss_value, average_abs_error = self._test(net, loss_function=loss_function)
        print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, average_abs_error))
        
    def compute_performance(self, net: torch.nn.Module, load_from_checkpoint: bool = True) -> tuple:
        """Get the performance metrics for an individual

        Args:
            net (torch.nn.Module): The model whose performance needs to be assesed
            load_from_checkpoint (bool, optional): Wether to load the weights from a previous workout session. Defaults to True.

        Returns:
            tuple: Loss value and accuracy of the model on the test/validation set
        """
        if load_from_checkpoint:
            checkpoint = torch.load(self.directory, map_location = self.device)
            net.load_state_dict(checkpoint['model_state_dict'])
        loss_function = self._get_loss()
        loss_value, average_abs_error = self._test(net, loss_function=loss_function)
        return loss_value, average_abs_error
    
    def show_performance(self, net: torch.nn.Module, load_from_checkpoint: bool = True) -> None:
        """Print performance info to screen

        Args:
            net (torch.nn.Module): The model
            load_from_checkpoint (bool, optional): Wether to load the weights from a previous workout session. Defaults to True.
        """
        loss_value, accuracy = self.compute_performance(net, load_from_checkpoint)
        print("\n------ PERFORMANCE ON TEST SET ------")
        print("Loss function value: {:.2f} \t Accuracy: {:.2f}%\n".format(loss_value, accuracy))
        
    def show_learning_curves(self, save_file: bool = False):
        checkpoint = torch.load(self.directory, map_location = self.device)
        results = checkpoint['results']
        epoch = checkpoint['epoch']
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 8))
        ax1.plot(np.arange(1, epoch+1), results["train_loss"], label = "train")
        ax1.plot(np.arange(1, epoch+1), results["val_loss"], label = "validation")
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss Function')
        ax1.legend()
        ax2.plot(np.arange(1, epoch+1), results["train_avg_abs_err"], label = "train")
        ax2.plot(np.arange(1, epoch+1), results["average_abs_error"], label = "validation")
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        if save_file:
            plt.savefig(self.directory + "my_plot.png")
        
        