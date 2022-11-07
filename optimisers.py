import numpy as np
import NeuralNet


class Optimiser():
    """Base optimiser for Neural Nets
    Any other optimiser should be subclass of this one, such that set_model(self, model) and update(self, derivatives) are callable
    """    
    
    def __init__(self, lr) -> None:
        self.lr = lr
        self.model = None
    
    def set_model(self, model):
        self.model = model
    
    def update_model(self, derivatives):
        '''
        input: derivatives is a list of lists, of the kind [[dbeta_W, dbeta_B], ...] which contains the change which must be 
        made to the parameters W and B.
        '''
        dbetas = self.compute_dbetas(derivatives)
        if self.model is None:
            raise ValueError("No model has been set for optimiser")
        for dbeta, layer in zip(dbetas, self.model.layers):
            layer.W -=  dbeta[0]
            layer.B -=  dbeta[1]
    
    def compute_dbetas(self, derivatives):
        ''' compute the changes to the parameters to be made, here using plain gd.
        input: derivatives is a list of lists, each of the kind [ dCdW , dCdB]
                dCdW, dCdB are np.array (or np.ndarray?)
        '''
        dbetas = []
        for derivative in derivatives:
            dbetas.append([  lr*derivative[0], lr*derivative[1]  ]  )



class MomentumOptimiser(Optimiser):
    
    def __init__(self, lr, momentum):
        self.carry = 1. - 1./momentum
        self.velocity = None
        super().__init__(lr)

    #do we really need this override if the function does the very same thing?? I comment it out 
    def set_model(self, model):
        super().set_model(model)
        #in this way, setting a new model will automatically set the velocities to zero.
        #no need for if test in long loops (gradient descent)
        self.velocity = [ [ np.zeros(shape=self.model.layer.W), np.zeros(shape=self.model.layer.B)] for layer in self.model.layers ]

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
        self.velocity = [ [self.carry * vel[0] + der[0].mean(axis = 2)[:,:,np.newaxis], \
        self.carry * vel[1] + der[1].mean(axis = 1)[:,np.newaxis]] for vel, der in zip(self.velocity, derivatives)]
        return self.velocity


class AdaGradOptimiser( Optimiser ):

    def __init__( self, lr, epsilon ):
        #initialize learning rate using superclass
        super().__init__(lr)
        #self.G will contain sum of gradients^2
        self.G = None
        self.epsilon = epsilon

    def set_model(self, model):

        super().set_model(model)
        #in this way, setting a new model will automatically set the sum of gradients to zero.
        self.G = [ [ np.zeros(shape=self.model.layer.W), np.zeros(shape=self.model.layer.B)] for layer in self.model.layers ]

    def compute_dbetas(self, derivatives):
        #first compute dbeta (at first step, self.G=0, so only need to update it after computaiton of dbeta)
        dbeta = [[der[0]*lr/ (np.sqrt(Gsum[0])+self.epsilon) , der[1]*lr/(np.sqrt(Gsum[1])+self.epsilon) ] \
        for Gsum, der in zip(self.G, derivatives)]
        #update G= sum of gradients squared
        self.update_G(derivatives)
        return dbeta

    def update_G( derivatives ):
        self.G = [[ G_single[0] + der[0]**2 , G_single[1]+der[1]**2 ] for G_single, der in zip(self.G, derivatives)]
        return

class RMSPropOptimiser( AdaGradOptimiser ):
    #note I make a subclass of AdaGradOptimiser, cause the architecture looks very similar
    def __init__(self, lr, epsilon, gamma):

        super().__init__(lr, epsilon)
        self.gamma = gamma

    def update_G( self, derivatives):
        #the computation of d_beta is the very same as in adagrad, now the quantity G is updated differently.
        self.G = [[ self.gamma*G_single[jj] + (1-self.gamma)*der[jj]**2  \
        for jj in range(2)] for G_single, der in zip(self.G, derivatives)]
        return


class AdamOptimiser ( AdaGradOptimiser ):

    def __init__(self, lr , epsilon, gamma1, gamma2):
        super().__init__(lr, epsilon)
        #now need two parameters to update 1st and 2nd order moments of gradients
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        #need also the gamma1^t , gamma2^t with t= nr. of steps done.
        self.power_of_gammas =  np.array([gamma1, gamma2])
        #here I will store the first moment of the gradient
        self.M = None

    def set_model(self, model):

        super().set_model(self, model):
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
        dbetas = []
        for m_single, g_single in zip(self.M, self.G):
            tmplist = []
            for ii in range(2):
                mhat, ghat = self.hat(np.array( [m_single[ii], g_single[ii] ] ) )
                tmplist.append( lr*mhat/(np.sqrt(ghat) + self.epsilon ) )
            dbetas.append( tmplist )
        #updates to M and G are done after computing dbetas, for at the first time we consider M, G = zero
        self.update_G( derivatives )

    def update_G( self, derivatives ):
        '''update G, M and powers of gamma1 and gamma2.'''

        Glist = []
        Mlist = []
        for m_single, g_single, der in zip(self.M, self.G, derivatives)
            tmplist_m = []
            tmplist_g = []
            for ii in range(2):
                der_ = der[ii]
                der2_ = der_**2
                tmplist_m.append(self.gamma1*m_single[ii] + (1-gamma1)*der_)
                tmplist_g.append(self.gamma2*g_single[ii] + (1-gamma2)*der2_ )
            Glist.append(tmplist_g)
            Mlist.append(tmplist_m)
        self.G = Glist
        self.M = Mlist

        #finally update powers of gamma1 and gamma2
        self.power_of_gammas*=self.power_of_gammas


