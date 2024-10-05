This directory contains the classes that define the model of the agents, which includes the following files:

1. ```network_structures.py```: This file defines the **feature extractor** of the models used by MaskablePPO (a class created by the module ```sb3_contrib```), MaskablePPOSingleCoach and MaskablePPOMultiCoach (modifications of MaskablePPO created by us).

2. ```mc_policy.py```: This file **combines** multiple policies (coaches) into a single policy that can be viewed as single-coach. By "combine" we mean to randomly select a coach, for each state, by the Softmax Distribution according to the value function of that state.