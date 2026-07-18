## Rolling Notes

#### Lower grid level

    # mf.grids.level = GRID_LEVEL
    mf.grids.level = 2


#### Resolve memory issues

    print(f"\n[{name}] SCF startowe...")
    mf = dft.RKS(mol)
    mf.max_memory = 12000 # zmiana

    print(f"\n[{name}] SCF na optymalnej geometrii...")
    mf_eq = dft.RKS(mol_eq)
    mf_eq.max_memory = 12000 # zmiana

#### Save screen output to file 

`$ python3 nmr_ch4_vs_tms.py 2>&1 | tee obliczenia_output.log`


#### Clear temporary cache

It may be necessary to clean the /tmp directory between consecutive program runs.

`$ rm -f /tmp/tmp*`


#### Update TMS coordinates

```tms_xyz = '''
Si       0.00000000    0.00000000    0.00000000
C        0.00000000    1.53500300    1.08541100
H        0.88796800    1.59368500    1.73177600
H       -0.88796800    1.59368500    1.73177600
H        0.00000000    2.42468300    0.44391600
C        0.00000000   -1.53500300    1.08541100
H        0.00000000   -2.42468300    0.44391600
H        0.88796800   -1.59368500    1.73177600
H       -0.88796800   -1.59368500    1.73177600
C        1.53500300    0.00000000   -1.08541100
H        2.42468300    0.00000000   -0.44391600
H        1.59368500    0.88796800   -1.73177600
H        1.59368500   -0.88796800   -1.73177600
C       -1.53500300    0.00000000   -1.08541100
H       -1.59368500    0.88796800   -1.73177600
H       -1.59368500   -0.88796800   -1.73177600
H       -2.42468300    0.00000000   -0.44391600
'''
