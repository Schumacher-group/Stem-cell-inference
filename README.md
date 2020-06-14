============ Stem cell inference code ============ 

Please note: 'Model X/Y' in the following Python files refers to model C/U as defined in the manuscript, respectively.

TS_Bayes.py
	Bayesian parameter and model inference based on time series (TS) data

DTO_X.py
	Define all functions needed for 'sample path' algorithm (trajectory sampling, likelihood calculation, 
	Poisson model sampling, Poisson model likelihood, plotting results, ...)

DTO_X_test_multi.py
	Main file for executing parameter inference algorithm for model X. Uses functions defined in DTO_X.py

Same structure for DTO_Y.py and DTO_Y_test_multi.py

DTO_MS.py
	Defines function for model comparison X<->Y. Bayes factor calculation using 1. Harmonic Mean, 
	2. Thermodynamic Intergration, 3. Product space search.

DTO_compare.py
	Main file for executing model selection algorithms. Uses functions defined in DTO_MS.py

EPISC_ABC_simple.py
	Defines functions for artificial data creation from extended Models X,Y, functions for reading and processing 
	experimental data and defines ABC inference algorithms (rejection sampling).

EPISC_data.py
	Main file for executing inference algorithms based on artificial or experimental data. 
	Also plots resulting Posteriors. Uses functions defined in EPISC_ABC_simple.py

EPISC_MCMC.py
	Older version of algorithm which tried exact inference for EpiSC model using the original MCMC algorithm.
	Worked less well because state space too large.
	
pyABC_runall.py
	Main file for reading and processing experimental data from extended Models X,Y
	and defines ABC inference algorithm (SMC-ABC) using the pyabc library.
	
