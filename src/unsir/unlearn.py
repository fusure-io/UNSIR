import torch
import numpy as np
import torch.nn as nn
from torch.data.utils import DataSet,DataLoader


class Noise(nn.Module):
    def __init__(self, *dim):
        super().__init__()
        self.noise = nn.Parameter(torch.randn(*dim), requires_grad = True)

    def forward(self):
        return self.noise

class NoisyDataset(Dataset):
    def __init__(self, noisy_data):
        self.noisy_data = noisy_data

    def __len__(self):
        return len(self.noisy_data)

    def __getitem__(self, idx):
        input_data, label = self.noisy_data[idx]
        input_data = input_data.view(3, 32, 32).to(torch.float32) 
        result = { 'image':input_data, 
                   'label':label
                 }
        return result

class Unlearn():
    """
    Algorithm to perform the forgetting.
    """

    def __init_(self,model:torch.nn.Module,
                forget_data,retain_set:DataSet,
                device:Str='cpu',
                is_dataset:Bool=False,
                batch_size:int=64,
                shape:tuple,
                dim:int):

        """
        model :the learned model
        forget_data: Can be a List of classes to forget or A dataset object of the forget set,
        retain_set: A Dataset of object of retain set ,
        device: cpu or gpu device. The default is cpu,
        is_dataset: Whether the forget_data is a dataset or a list. The default is False,
        batch_size:int - Prefered batch size. The default is 64,
        shape:tuple - Shape of the input that the model expects,
        dim:int - The dimentions of the input

        """
        if is_dataset==True:
            self.forget_classes = list(set([forget_data.__getitem__(i)['target'] for i in range(forget_data.__len__())]))
        else:
            if forget_data:
                if  type(forget_data) =='list':
                    if  len(list) !=0:
                        self.forget_classes=forget_data
                    else:
                        raise Exception("Please pass in a non empty list")
                else:
                    raise TypeError("The forget_data item has to be list")
            else:
                raise Exception("Please pass int one of:\n 1. Forget set\n 2.Forget classes")
        self.model=model
        self.retain_set=retain_set
        self.device=device
        self.dim=dim
        self.shape=shape
        self.batch_size=batch_size 
        self.retain_loader = DataLoader(self.retain_set,batch_size=self.batch_size)
        
    def error_maximizing_noise(self) -> Dataset:
        """
        batch_size: The prefered batch size
        shape: shape of the input
        dim: dimention of the input. Expected dimensions are 3 for colored images & 2 for black and white
        """
        
        if self.dim == 3:
            B,T,C = self.shape
            mean_dim=[1, 2, 3]
        else:
            T,c=self.shape
            mean_dim=[1, 2]

        noises = {}

        for cls in self.forget_classes:
            print("Optiming loss for class {}".format(cls))
            if dim ==3:
                noises[cls] = Noise(self.batch_size, B, T, C).to(self.device)
            else:
                noises[cls] = Noise(self.batch_size, T, C).to(self.device)

            opt = torch.optim.Adam(noises[cls].parameters(), lr = 0.1)

        num_epochs = 5
        num_steps = 8
        class_label = cls

        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls]()
                labels = torch.zeros(self.batch_size).to(self.device)+class_label
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = -nn.functional.cross_entropy(outputs, labels.long())+ 0.1*torch.mean(torch.sum(torch.square(inputs), mean_dim))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            print("Loss: {}".format(np.mean(total_loss)))

        noisy_data = []
        num_batches = 20

        for cls in self.forget_classes:
            for i in range(num_batches):
                batch = noises[cls]().cpu().detach()
                for i in range(batch[0].size(0)):
                    noisy_data.append((batch[i], cls))

        noisy_dataset = NoisyDataset(noisy_data)

        return noisy_dataset

    def impair(self,impair_steps=1,learning_rate=0.1,momentum:float=0.9,weight_decay:float=5e-4) -> torch.nn.Module:
        """
        A single pass of the impair step.

        impair_steps: Integer. The number of timer the model trains on the noise and retain dataset
        learning_rate: Float. The default is 0.1, 
        momentum=momentum:Float. The default is 0.9, 
        weight_decay:Float. The default is 5e-4
        """
        try:
            noisy_dataset = self.error_maximizing_noise()
            noisy_loader = DataLoader(noisy_dataset,batch_size=self.batch_size)
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=momentum)

            for epoch in range(impair_steps):
                self.model.train(True)  # Set the model to training mode
                running_loss = 0.0
                running_acc = 0
                # Iterate over both noisy data and retain data loaders in an alternating fashion
                for i, (noisy_data, retain_data) in enumerate(zip(noisy_loader, self.retain_loader)):
                    # Process noisy data
                    noisy_inputs = noisy_data["image"]
                    noisy_labels = noisy_data["label"]
                    noisy_inputs, noisy_labels = noisy_inputs.to(self.device), noisy_labels.to(self.device)

                    optimizer.zero_grad()
                    noisy_outputs = self.model(noisy_inputs)
                    noisy_loss = nn.functional.cross_entropy(noisy_outputs, noisy_labels)
                    noisy_loss.backward()
                    optimizer.step()


                    retain_inputs = retain_data["image"]
                    retain_labels = retain_data["label"]
                    retain_inputs, retain_labels = retain_inputs.to(self.device), retain_labels.to(self.device)

                    optimizer.zero_grad()
                    retain_outputs = self.model(retain_inputs)
                    retain_loss = nn.functional.cross_entropy(retain_outputs, retain_labels)
                    retain_loss.backward()
                    optimizer.step()

                    combined_loss = noisy_loss + retain_loss
                    running_loss += combined_loss.item()
                    out = torch.argmax(noisy_outputs.detach(), dim=1)
                    assert out.shape == noisy_labels.shape
                    running_acc += (noisy_labels == out).sum().item()

                # Print statistics for the epoch
                print("Successfully ran the Impair Step")
                return "Success"
        except Exception as e:
            print(str(e))
            return e

    def repair(self,
               repair_steps=1
               learning_rate:float=0.1, 
               momentum:float=0.9,
               weight_decay:float=5e-4):
        """
        A single pass of the repair step.

        repair_steps: Integer. The number of timer the model trains on the retain dataset
        learning_rate: Float. The default is 0.1, 
        momentum=momentum:Float. The default is 0.9, 
        weight_decay:Float. The default is 5e-4
        """
        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

            self.model.train()

            for _ in range(repair_steps):
                for sample in self.retain_loader:
                    inputs = sample["image"]
                    targets = sample["age_group"]
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
            print("Successfully ran the Impair Step")
            return "Success"
        except Exception as e:
            print(str(e))
            return str(e)