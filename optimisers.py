import numpy as np
import NeuralNet


class Optimiser():
    """Base optimiser for Neural Nets
    
        Any other optimisers should be subclasses of this one, such that set_model(self, model) and update(self, derivatives) are callable
    """    
    
    
    def __init__(self, lr) -> None:
        self.lr = lr
        self.model = None
    
    def set_model(self, model):
        self.model = model
    
    def update(self, derivatives):
        if self.model is None:
            raise ValueError("No model has been set for optimiser")
        for derivative, layer in zip(derivatives, self.model.layers):
            layer.W -= self.lr * derivative[0].mean(axis = 2)
            layer.B -= self.lr * derivative[1].mean(axis = 1)[:,np.newaxis]
            
class MomentumOptimiser(Optimiser):
    
    def __init__(self, lr, momentum):
        self.carry = 1. - 1./momentum
        self.velocity = None
        super().__init__(lr)
        
    def set_model(self, model):
        super().set_model(model)
        
    def update(self, derivatives):
        if self.velocity is None:
            self.velocity = derivatives
            for vel in self.velocity:
                vel[0] = vel[0].mean(axis = 2)[:,:,np.newaxis]
                vel[1] = vel[1].mean(axis = 1)[:,np.newaxis]
        else:
            for vel, der in zip(self.velocity, derivatives):
                vel[0] = self.carry * vel[0] + der[0].mean(axis = 2)[:,:,np.newaxis]
                vel[1] = self.carry * vel[1] + der[1].mean(axis = 1)[:,np.newaxis]
        
        super().update(self.velocity)  
    