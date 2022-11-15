import numpy as np
import NeuralNet
from numba import jit
from numba.experimental import jitclass

class Optimiser():
    """Base optimiser for Neural Nets
    Any other optimiser should be subclass of this one, such that set_model(self, model) and update(self, derivatives) are callable
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

    def update_model(self, derivatives) -> None:
        """Do one step of gradient descent using the derivatives given by the model

        Args:
            derivatives (list of tuples of dCd_): Derivatives of all of the values in the model

        Raises:
            ValueError: If no medel has been set
        """
        dbetas = self.compute_dbetas(self.add_regularisation(derivatives))
        if self.model is None:
            raise ValueError("No model has been set for optimiser")
        for dbeta, layer in zip(dbetas, self.model.layers):
            layer.W -= dbeta[0]
            layer.B -= dbeta[1]

    def add_regularisation(self, dbetas):
        return [[dbeta[0] + self.lamda * layer.W[:,:,np.newaxis],
                 dbeta[1] + self.lamda * layer.B                 ]
                for dbeta, layer in zip (dbetas, self.model.layers)]

    def compute_dbetas(self, derivatives):
        ''' compute the changes to the parameters to be made, here using plain gd.
        input: derivatives is a list of lists, each of the kind [ dCdW , dCdB]
                dCdW, dCdB are np.array (or np.ndarray?)
        '''
        dbetas = []
        for derivative in derivatives:
            dbetas.append([  self.lr*derivative[0].mean(axis=2), self.lr*derivative[1].mean(axis=1)[:,np.newaxis]  ]  )

        return dbetas



class MomentumOptimiser(Optimiser):

    def __init__(self, lr, momentum = 1, lamda = 0) -> None:
        self.carry = 1. - 1./momentum
        self.velocity = None
        super().__init__(lr, lamda)

    def set_model(self, model):
        super().set_model(model)
        #in this way, setting a new model will automatically set the velocities to zero.
        #no need for if test in long loops (gradient descent)
        self.velocity = [ [ np.zeros(shape=layer.W.shape), np.zeros(shape=layer.B.shape)] for layer in self.model.layers ]

    ##note by gianmarco: commenting out old update version, now the update function changed its name
    ##to "update_model" and does not need to be
    ##over ridden, the only thing which needs

    # def update(self, derivatives):
    #     #at the first step, simply compute gradient descent step using plain gradient.
    #     if self.velocity is None:
    #         self.velocity = derivatives

    #     #at every step use momentum based gd to compute gd step.
    #     else:#####PROBLEM: THIS ELSE DOES NOT REALLY UPDATE ANYTHING(!?)
    #         for vel, der in zip(self.velocity, derivatives):
    #


    #     #doubt about the following line: self.velocity at this point should be still be velocity before the else statement
    #     super().update(self.velocity)
    #
    def compute_dbetas(self, derivatives):
        self.velocity = [ [self.lr*(self.carry * vel[0] + der[0].mean(axis = 2)),
                           self.lr*(self.carry * vel[1] + der[1].mean(axis = 1)[:,np.newaxis])]
                         for vel, der in zip(self.velocity, derivatives)]
        return self.velocity


class AdaGradOptimiser( Optimiser ):

    def __init__( self, lr, epsilon, lamda=0):
        #initialize learning rate using superclass
        super().__init__(lr, lamda)
        #self.G will contain sum of gradients^2
        self.G = None
        self.epsilon = epsilon

    def set_model(self, model):

        super().set_model(model)
        #in this way, setting a new model will automatically set the sum of gradients to zero.
        self.G = [ [ np.ones(shape=layer.W.shape)*self.epsilon,
                     np.ones(shape=layer.B.shape)*self.epsilon]
                  for layer in self.model.layers ]

    def compute_dbetas(self, derivatives):
        #update G= sum of gradients squared
        self.update_G(derivatives)
        #first compute dbeta (at first step, self.G=0, so only need to update it after computaiton of dbeta)
        dbeta = [[der[0].mean(axis = 2)              *self.lr/ (np.sqrt(Gsum[0])),
                  der[1].mean(axis = 1)[:,np.newaxis]*self.lr/ (np.sqrt(Gsum[1])) ]
        for Gsum, der, layer in zip(self.G, derivatives, self.model.layers)]
        return dbeta

    def update_G(self, derivatives):
        self.G = [[ G_single[0] + der[0].mean(axis=2)**2,
                    G_single[1] + der[1].mean(axis=1)[:,np.newaxis]**2 ]
                  for G_single, der in zip(self.G, derivatives)]
        return

class RMSPropOptimiser( AdaGradOptimiser ):
    #note I make a subclass of AdaGradOptimiser, cause the architecture looks very similar
    def __init__(self, lr, epsilon=1e-8, gamma=0.9, lamda=0):

        super().__init__(lr, epsilon, lamda)
        self.gamma = gamma

    def update_G( self, derivatives):
        #the computation of d_beta is the very same as in adagrad, now the quantity G is updated differently.
        self.G = [[ self.gamma*G_single[0] + (1-self.gamma)*der[0].mean(axis=2)**2,
                    self.gamma*G_single[1] + (1-self.gamma)*der[1].mean(axis=1)[:,np.newaxis]**2,]
                  for G_single, der in zip(self.G, derivatives)]
        return


class AdamOptimiser ( AdaGradOptimiser ):

    def __init__(self, lr , epsilon=1e-8, gamma1=0.9, gamma2=0.999, lamda=0):
        super().__init__(lr, epsilon, lamda)
        #now need two parameters to update 1st and 2nd order moments of gradients
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        #need also the gamma1^t , gamma2^t with t= nr. of steps done.
        self.power_of_gammas =  np.array([gamma1, gamma2])[:,np.newaxis,np.newaxis]
        #here I will store the first moment of the gradient
        self.M = None

    def set_model(self, model):

        super().set_model(model)
        #also create space to store first order moment of gradient,
        #initialized as zero. This class is subclass of AdaGradOptimiser, so self.G has been already properly initialized
        self.M = self.G.copy()

    def hat(self, x):
        '''function that puts the hat on M and G, dividing them by 1-gamma_1^t or 1-gamma_2^t.
        This function will act on the vectors M and G element-wise.
        inputs:
        x: np.array of the form [m, g]
        '''
        x/(1-self.power_of_gammas)
        return (x[0], x[1])

    def compute_dbetas(self , derivatives ):
        '''compute update to the parameters using ADAM optimization method.'''
        #updates to M and G are done after computing dbetas, for at the first time we consider M, G = zero
        self.update_G( derivatives )
        dbetas = []
        for m_single, g_single in zip(self.M, self.G):
            tmplist = []
            for ii in range(2):
                mhat, ghat = self.hat(np.array( [m_single[ii], g_single[ii] ] ) )
                tmplist.append(self.lr*mhat/(np.sqrt(ghat) + self.epsilon ) )
            dbetas.append( tmplist )

        return dbetas

    def update_G( self, derivatives ):
        '''update G, M and powers of gamma1 and gamma2.'''

        Glist = []
        Mlist = []
        for m_single, g_single, der in zip(self.M, self.G, derivatives):
            tmplist_m = []
            tmplist_g = []
            der_ = [der[0].mean(axis=2), der[1].mean(axis=1)[:,np.newaxis]]
            for ii in range(2):
                der2_ = der_[ii]**2
                tmplist_m.append(self.gamma1*m_single[ii] + (1-self.gamma1)*der_[ii])
                tmplist_g.append(self.gamma2*g_single[ii] + (1-self.gamma2)*der2_ )
            Glist.append(tmplist_g)
            Mlist.append(tmplist_m)
        self.G = Glist
        self.M = Mlist

        #finally update powers of gamma1 and gamma2
        self.power_of_gammas*=self.power_of_gammas

# @jitclass
class LrScheduleOptimiser():

    def __init__(self, lr_func, optimiser) -> None:
        self.lr_func = lr_func
        self.optimiser = optimiser
        self.count = 0

    def set_model(self, model) -> None:
        self.optimiser.set_model(model)

    def update_model(self, derivatives) -> None:
        self.count += 1
        self.optimiser.lr = self.lr_func(self.count)
        self.optimiser.update_model(derivatives)
