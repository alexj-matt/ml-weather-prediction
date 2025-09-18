# ML Project README

All julia code is located in "rain.jl" and we recommend VScode to execute the program, since it is more flexible and faster than pluto notebooks. The program loads and treats the supplied csv files "testdata.csv" and "trainingdata.csv", located in "./res/".

For each machine there are two code block:

1. *Fitting:*     The first code block is used to define a model and fit a machine using the model on the treated training data. The fitted machine is stored to "./machines/mach_i.jlso" (for the ith machine).
2. *Predicting:*  In the second code block the machine is used to predict given a treated testset. The resulting predicitons are stored in "./out/output_i.csv" (for the ith machine).

**A) Reproduce results:** To reproduce the outputs given in "./out/" after an initial execution of the first two code blocks, the *fitting* and *predicting* code block the desired machine can be run.
For machines where the corresponding .jlso file is already present it is not necessary to run the *fitting* code block, when given that the treated training data and the model have not been changed.

**B) New test data:** When wanting to predict on a different set of observations, it is best to replace "testdata.csv" by a file (identically named) containing the new observations and run the "read & treat" code block in "rain.jl". Then the procedure as in **A)** can be applied.

The figures produced from the treated training data are stored in "./figures/".

