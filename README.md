# SECS4M
PENDING PUBLICATION 
AUTHOR: ODYSSEAS VLACHOPOULOS
This is an AI-based surrogate model for the EC-JRC crop growth model ECroPS for grain maize. It utilizes 3 weather variables (min and max temperature and precipitation) in order to predict the Total Weight of Storage Organs (yield).
The nested RNN has been trained with native resolution of ERA5, and works under the assumption of maximum resolution that of ERA5 (~25km).
The functioning of the RNN as described within is important to understand the shape of the required input and the output, as well as the generation of the input feature space.
It generates predictions per batch, assuming feeding streams of daily weather from a given sowing date to the 26th of December, same as the ECroPS simulation does.
The surrogate model, just like ECroPS works per cell, meaning that each run is independent. It inherently retains the parallelizability and the distribution capacities required for large scale and/or ensemble simulations.
Also the code to generate Areas of Concern (AoC) from 3 different sources of climate is included:
ERA5, ECMWF SEAS5.1 and CMIP6 high resolution historical and projections. See inside for details.
