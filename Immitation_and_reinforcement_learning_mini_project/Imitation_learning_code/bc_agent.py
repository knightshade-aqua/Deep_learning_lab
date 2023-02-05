import torch
from agent.networks import CNN

class BCAgent:
    
    def __init__(self):
        # TODO: Define network, loss function, optimizer
        self.net = CNN()
        self.net.cuda()
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(),lr=1e-4)
        #pass

    def update(self, X_batch, y_batch):
        # TODO: transform input to tensors
        # TODO: forward + backward + optimize
        #X_batch = torch.tensor(X_batch)
        #y_batch = torch.tensor(y_batch)
        # TODO: forward + backward + optimize
        y_batch = y_batch.type(torch.LongTensor).cuda()
        output = self.net(X_batch)
        loss = self.loss_function(output, y_batch)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, output

    def predict(self, X):
        # TODO: forward pass
        X = X.cuda()
        outputs = self.net(X)
        return outputs

    def load(self, file_name):
        self.net.load_state_dict(torch.load(file_name))


    def save(self, file_name):
        torch.save(self.net.state_dict(), file_name)

