## BSIM-CMG model (DC only) in Python

Updated: 3/28/2018

Python version: v3.6.4

This is an attempt to write a working BSIM-CMG model in Python. Currently DC and RDSMOD = 0 only.

## Usage
Step 1: Add or modify model and instance parameters in 'modelcard.l'

Step 2: Run 'test.py' and see results.

Note: You can compare the results with commercial simulators like HSPICE.

Please help me debug this tool. Send feedback to `huanlinberkeley@gmail.com`

## Example modelcard.l
vd = 1.0  
vg = 1.0  
vs = 0.0  
vb = 0.0  
temp = 27.0  
L = 16e-9  
NFIN = 4  
VSAT = 125000  
U0 = 0.025  
DVTSHIFT = 0.01  

## test.py Results
Id =  3.592760184e-04 A  
Ig =  0.000000000e+00 A  
Is = -3.592760184e-04 A  
Ib =  0.000000000e+00 A





## BSIM-CMG model (DC only, with gradients) in Python

Update 9/20/2020 by Colin Hart

TensorFlow v. 2.2.0

This is a simple extension to the existing work by Huan Lin that takes advantage of automatic differentiation capabilities in Google's TensorFlow to provide gradients for the BSIM model.  These gradients are needed when solving for the DC operating point of a network containing nonlinear circuits [Vladimirescu 1994].

The operating point voltages must be removed from the model card and passed in as tensorflow variables.  The code in `bsimcmg.py` is unchanged. 

gm = Id/vg is used as an example in , but all 16 derivitives combining each device current with each terminal voltage can be retrieved in this way.

## Example modelcard_without_op.l

temp = 27.0  
L = 16e-9  
NFIN = 4  
VSAT = 125000  
U0 = 0.025  
DVTSHIFT = 0.01  

## test_gradients.py Results

Id =  3.592760184e-04 A  
Ig =  0.000000000e+00 A  
Is = -3.592760184e-04 A  
Ib =  0.000000000e+00 A
gm = 1.375134569e+00 mS
