Influence maximization in Boolean networks
=======================================================

Code to generalize methods (e.g. mean-field approximation) for influence maximization to Boolean networks.

**Tutorials:**

<p>Tutorial_Drosophila_SPN.ipynb shows how to run the individual based mean-field approximation (IBMFA), calculate entropy, and retrieve driver sets for biological networks (e.g. the _drosophila_ single-cell SPN).</p>
<p>Tutorial_RBN.ipynb shows how to run the IBMFA, calculate entropy, and retrieve driver sets for random Boolean networks (RBNs).</p>


**Scripts:**

<p>mean_field_computations.py (code to run IBMFA)</p>
<p>entropy_computations.py (code to calculate entropy of network configurations)</p>
<p>driver_sets.py (code to find driver sets towards fixed points of a network)</p>
<p>simulations.py (code to run and analyze simulations)</p>
<p>brute_force_computations.py (code to run and analyze brute-force calculations for small networks)</p>
<p>RBN_computations.py (code for additional functions needed for RBNs)</p>
<p>modules.py (utility functions related to influence pathways and code for general threshold networks)</p>
<p>utils.py (utility functions for the IBMFA)</p>

**Original notebooks:**

<p>Note: the scripts above were created from the functions originally used in the jupyter notebooks below.  If there's a bug in the above code, you may refer to the original function defined in the notebook to help troubleshoot.</p>
<p>See information_diffusion.ipynb for results on specific genetic regulatory networks (GRNs).</p>
<p>See dynamic_game_theory.ipynb for results on random Boolean networks (RBNs) and the Cell Collective repository (http://cellcollective.org/) [1]</p>
<p>See Mean-field Approximation in Boolean Networks.ipynb for figure results used in the paper.</p>
<p>See network_attractors.ipynb for results on attractors and control kernels.  These calculations require code from https://zenodo.org/record/5172898 [2]</p>


<p>The corresponding paper has been submitted for publication.</p>

References:
---------

[1] Tom´aˇs Helikar, Bryan Kowal, Sean McClenathan, Mitchell Bruckner, Thaine Rowley, Alex Madrahimov, Ben Wicks, Manish Shrestha, Kahani Limbu, and Jim A
Rogers. The cell collective: toward an open and collaborative approach to systems biology. BMC systems biology, 6(1):1–14, 2012.
[2] Borriello, E., Daniels, B.C. The basis of easy controllability in Boolean networks. Nat Commun 12, 5227 (2021). https://doi.org/10.1038/s41467-021-25533-3
