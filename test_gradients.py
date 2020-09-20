import re
from bsimcmg import *

import tensorflow as tf

def read_mdl(file):
    mdl = {}
    with open(file,'r') as f:
        lines = f.read().splitlines()
    for line in lines:
        param, value = re.split('[=\s]+', line)
        mdl[param] = float(value)
    return mdl


filepath = "modelcard_without_op.l"
param = read_mdl(filepath)

vd = tf.Variable(1.0)
vg = tf.Variable(1.0)
vs = tf.Variable(0.0)
vb = tf.Variable(0.0)

param['vd'] = vd
param['vg'] = vg
param['vs'] = vs
param['vb'] = vb

# Newton iteration on DC model requires gradients.  
# Use tf autodiff to get these for free
with tf.GradientTape() as tape:
	Id, Ig, Is, Ib = BSIMCMG(**param).calc()

print(f'Id = {Id:>16.9e} A')
print(f'Ig = {Ig:>16.9e} A')
print(f'Is = {Is:>16.9e} A')
print(f'Ib = {Ib:>16.9e} A')

# dId_dvg as an example
# 16 combinations can be retrieved in the same way
gm = tape.gradient(Id, vg)

print(f'gm = {gm.numpy()*1000:>16.9e} mS')

'''
Id =  3.592760768e-04 A
Ig =  0.000000000e+00 A
Is = -3.592760768e-04 A
Ib =  0.000000000e+00 A
gm =  1.375134569e+00 mS
'''

