import numpy as np
import NeuralNet


class Optimiser():
    """Base optimiser for Neural Nets
    
        Any other optimisers should be subclasses of this one, such that set_model(self, model) and update(self, derivatives) are callable
    """    
    
    def __init__(self, lr, lamda = 0) -> None:
        self.lr = lr
        self.lamda = lamda
        self.model = None
    
    def set_model(self, model) -> None:
        if self.model is None:
            self.model = model
        else:
            raise ValueError("Model has already been set, Optimisers should not be reused")
    
    def update(self, derivatives) -> None:
        """Do one step of gradient descent using the derivatives given by the model

        Args:
            derivatives (list of tuples of dCd_): Derivatives of all of the values in the model

        Raises:
            ValueError: If no medel has been set
        """        
        if self.model is None:
            raise ValueError("No model has been set for optimiser")
        for derivative, layer in zip(derivatives, self.model.layers):
            layer.W -= self.lr * (derivative[0].mean(axis = 2) + self.lamda * layer.W)
            layer.B -= self.lr * (derivative[1].mean(axis = 1)[:,np.newaxis] + self.lamda * layer.B)
            
class MomentumOptimiser(Optimiser):
    
    def __init__(self, lr, lamda = 0, momentum = 1) -> None:
        self.carry = 1. - 1./momentum
        self.velocity = None
        super().__init__(lr = lr, lamda = lamda)
        
    def set_model(self, model) -> None:
        super().set_model(model)
        
    def update(self, derivatives) -> None:
        if self.model is None:
            raise ValueError("No model has been set for optimiser")
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
    
    
class LrScheduleOptimiser(Optimiser):
    
    def __init__(self, lr_func, lamda=0) -> None:
        self.count = 0
        self.lr_func = lr_func
        
        super().__init__(lr_func(0), lamda)   
        
    def update(self, derivatives) -> None:
        self.count += 1
        self.lr = self.lr_func(self.count)        
        super().update(derivatives)