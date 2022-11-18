#test of the gradient descent methods for the LinearModel class.

import warnings
import numpy as np
import LinearModel as lm
import optimisers as op
import matplotlib.pyplot as plt   

def testGD( optimiser, npoints=200, Nbatches=1, Nepochs=100, tolerance=1e-8, m=10, q=-1, noise=0.2):
	'''test gradient descent methods, using a linear regression to a straight line as an example.
		Note that the default values here are overridden by another function definition (generate_linreg_results),
		which gives other defaults.
	args:
		npoints: number of points in the training set
		optimiser: instance of an optimiser class. 
		Nbatches: number of batches used for SGD, defaults to 1.
		Nepochs: how many loops through the batches in SGD
		tolerance: stops gd when |Cost[ii+1]-Cost[ii]|<tolerance
		m: slope of line that generates data
		q: intercept of line that generates data
		noise: std of gaussian noise

	returns: (cost, beta0, beta1), tuple of np.ndarrays of shape (nsteps, 1)'''
	np.random.seed(15112022)
	print(f"\n\nRunning GD with {optimiser.name} optimiser")
	#generation of dataset
	x = np.random.rand(npoints,1)
	y = m*x + q + noise*np.random.randn(npoints,1)
	X = np.concatenate(  ( np.ones(x.shape) , x ) , axis=1  )
	#initialize model
	betas = lm.fakeLayer(2)
	model = lm.Model(betas, optimiser)

	#initialize empty lists for storing results
	cost = []
	beta0= []
	beta1= []
	beta0.append(model.layers[0].B[0,0])
	beta1.append(model.layers[0].B[1,0])

	#determine batch size and ensures all batches will have the same size
	if (npoints%Nbatches!=0):
		raise ValueError("The number of batches does not divide the number of points")
	
	batch_size = npoints//Nbatches
	print(f"npoints = {npoints}, Nbatches={Nbatches}")
	print(f"batch_size: {batch_size}")
	#initialize gd parameters 
	epoch = 1
	dCost = 10
	cost_ = 0
	gdsteps = 0
	idx_list = np.arange(npoints)

	#initialize rng to reshuffle data, seed it to ensure reproducibility of our results
	rng = np.random.default_rng(seed=20062002)

	#start gd
	while epoch<=Nepochs and dCost>tolerance:
		#shuffle points in the dataset to introduce stochasticity at every epoch
		rng.shuffle(idx_list)
		X = X[idx_list,:]
		y = y[idx_list]

		for batch_index in range(Nbatches):
			start_index = batch_index*batch_size
			dataset_batch = {'designmat':X[start_index:start_index+batch_size,:], 'target': y[start_index:start_index+batch_size]}
			oldCost = cost_
			cost_, beta_  = model.gd_step(dataset_batch)
			#update variation of cost function
			#if cost_ happens to be smaller than tolerance, will break at first iteration
			dCost = np.abs( cost_ - oldCost  )

			


			#save results for later analysis
			cost.append(cost_[0])
			beta0.append(beta_[0,0])
			beta1.append(beta_[1,0])

			if dCost < tolerance:
				break

			gdsteps+=1

		epoch+=1

	print(f"found the following parameters:\n beta0 = {beta0[-1]}, \n beta1={beta1[-1]}")
	print(f"cost function at last step = {cost[-1]}")

	#raise warning if convergence not reached
	if dCost > tolerance:
		warnings.warn("GD did not converge.")
		print(f"CONVERGENCE NOT REACHED. tolerance was {tolerance}. Change in cost after {gdsteps} was {dCost}")
		
	else:
		print(f"converged after {gdsteps} GD steps.")

	#want to return arrays and not lists for plotting purpose
	cost = np.array(cost).reshape(-1,1)
	beta0 = np.array(beta0).reshape(-1,1)
	beta1 = np.array(beta1).reshape(-1,1)

	return cost, beta0, beta1

def generate_linreg_results(optimiserlist, npoints = 1000, Nbatches=1, m=10, q=-1, noise=0.2, tol=1e-8, Nepochs=100):
	'''generate the results for a linear regression to a straight line y(x), with different
		optimization methods. If no convergence is reached, unexpected behaviour might happen.
		
		args:
			optimiserlist: list of Optimiser objects, already initialized
			npoints: number of points in dataset
			Nepochs: number of epochs used
			Nbatches: nr. of batches for sgd
			m= slope of line
			q= intercept of line
			noise = std of gaussian noise in data
			tol = tolerance for convergence
		returns: 
			results: list of np.ndarray, each of shape (steps, 3), where
			steps for result[ii] is the number of steps taken to converge by the gradient descent method
			defined in optimiserlist[ii].

	'''

	results = []

	for optimiser in optimiserlist:

		costs, beta0, beta1 = testGD(optimiser, npoints=npoints, Nbatches=Nbatches, Nepochs=Nepochs, m=m, q=q, noise=noise, tolerance=tol)
		current_result = np.concatenate((beta0, beta1), axis=1)

		results.append([costs, current_result])

	return results



def plot_gd_trajectories(results, optimiserlist):
	
	fig_traj, ax_traj = plt.subplots(1,1, figsize=(3.313, 5))

	for idx, result in enumerate(results):
		#note in label the last 9 charcacter of the .__name__ string are removed
		#they correspond to the word Optimiser.
		ax_traj.plot(result[1][:,0], result[1][:,1], label=optimiserlist[idx].name )

	ax_traj.grid(visible=True)
	ax_traj.set_xlabel("$\\hat \\beta_0$")
	ax_traj.set_ylabel("$\\hat \\beta_1$")
	ax_traj.legend(fontsize=10, loc="upper right")

	plt.show()

	return fig_traj, ax_traj

if __name__ == "__main__":

	optimiserlist = [op.Optimiser(lr=1),
	# op.MomentumOptimiser(lr=0.1, momentum=4),
	# op.AdaGradOptimiser(lr=0.1),
	# op.RMSPropOptimiser(lr=0.1),
	# op.AdamOptimiser(lr=0.1) 
	]

	results = generate_linreg_results(optimiserlist, npoints = 100, Nbatches=50 , Nepochs=1, noise=0.1, tol=1e-8)

	#plot_gd_trajectories(results, optimiserlist)


