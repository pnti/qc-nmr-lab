import numpy as np
from pyscf import gto, dft
from pyscf.prop import nmr
from pyscf.geomopt import geometric_solver
import time

# POPRAWKA DLA NOWEGO NUMPY:
#import pyscf.dft.numint
#pyscf.dft.numint.BLKSIZE = 256
#pyscf.dft.numint.BLKSIZE = 192
# POPRAWKA DLA NOWEGO NUMPY:
#import pyscf.dft.numint
#pyscf.dft.numint.__dict__['BLKSIZE'] = 192

# POPRAWKA DLA NOWEGO NUMPY I MODUŁU SGX:
import pyscf.dft.numint
import pyscf.sgx.sgx_jk
pyscf.dft.numint.BLKSIZE = 192
pyscf.sgx.sgx_jk.BLKSIZE = 192

XC = 'B3LYP'
BASIS = 'pcSseg-2'
VERBOSE = 4
GRID_LEVEL = 3

def run_nmr(mol_xyz, name, charge=0, spin=0):
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"START: {name}")
    print(f"{'='*50}")

    mol = gto.Mole()
    mol.atom = mol_xyz
    mol.basis = BASIS
    mol.unit = 'Angstrom'
    mol.charge = charge
    mol.spin = spin
    mol.verbose = VERBOSE
    mol.build()
    print(f"Atomy: {mol.natm}, elektrony: {mol.tot_electrons()}")

    print(f"\n[{name}] SCF startowe...")
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF nie zbiegło dla {name}")

    print(f"\n[{name}] Optymalizacja geometrii...")
    mol_eq = geometric_solver.optimize(mf, maxsteps=100)
    mol_eq.tofile(f'{name}_opt.xyz')

    print(f"\n[{name}] SCF na optymalnej geometrii...")
    mf_eq = dft.RKS(mol_eq)
    mf_eq.xc = XC
    mf_eq.grids.level = GRID_LEVEL
    mf_eq.conv_tol = 1e-10
    mf_eq.kernel()
    if not mf_eq.converged:
        raise RuntimeError(f"SCF po opt nie zbiegło dla {name}")

    print(f"\n[{name}] Liczenie GIAO-NMR...")
    #nmr_obj = nmr.NMR(mf_eq) # TUTAJ ZMIANA
    nmr_obj = nmr.rks.NMR(mf_eq)
    #nmr_obj.gauge_orig = 'giao'
    nmr_obj.kernel()

    shielding_tensors=nmr_obj.shielding()

    c_indices = [i for i, a in enumerate(mol_eq.elements) if a == 'C']
    sigma_c = []
    for idx in c_indices:
        tensor = shielding_tensors[idx]
        sigma_iso = np.trace(tensor) / 3.0
        sigma_c.append(sigma_iso)
        print(f"[{name}] C{idx}: σ_iso = {sigma_iso:.3f} ppm")

    t1 = time.time()
    print(f"[{name}] Gotowe w {t1-t0:.1f} s")
    return np.array(sigma_c), mol_eq

ch4_xyz = '''
C 0.000000 0.000000 0.000000
H 0.629000 0.629000 0.629000
H -0.629000 -0.629000 0.629000
H -0.629000 0.629000 -0.629000
H 0.629000 -0.629000 -0.629000
'''

tms_xyz = '''
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

print(f"Metoda: {XC}/{BASIS}")
print("Liczę CH4 i TMS w jednej sesji...")

systems = {'CH4': ch4_xyz, 'TMS': tms_xyz}
results = {}
for name, xyz in systems.items():
    sigma_c, mol_eq = run_nmr(xyz, name)
    results[name] = {
        'sigma_c_all': sigma_c,
        'sigma_c_avg': np.mean(sigma_c),
        'sigma_c_std': np.std(sigma_c),
        'mol': mol_eq
    }

print(f"\n{'='*50}")
print("PODSUMOWANIE")
print(f"{'='*50}")
sigma_ch4 = results['CH4']['sigma_c_avg']
sigma_tms = results['TMS']['sigma_c_avg']
delta = sigma_tms - sigma_ch4

print(f"Metoda: {XC}/{BASIS}")
print(f"σ_iso C CH4: {sigma_ch4:.3f} ppm")
print(f"σ_iso C TMS: {sigma_tms:.3f} ± {results['TMS']['sigma_c_std']:.3f} ppm")
print(f"δ 13C CH4 vs TMS: {delta:.2f} ppm")
print(f"Eksperyment: -2.3 ppm")
print(f"Błąd: {delta - (-2.3):.2f} ppm")

rozrzut_tms = max(results['TMS']['sigma_c_all']) - min(results['TMS']['sigma_c_all'])
print(f"\nRozrzut C w TMS: {rozrzut_tms:.3f} ppm | powinno być <0.1 ppm")

with open('wynik_NMR.txt', 'w') as f:
    f.write(f"Metoda: {XC}/{BASIS}\n")
    f.write(f"sigma_CH4 = {sigma_ch4:.3f} ppm\n")
    f.write(f"sigma_TMS = {sigma_tms:.3f} ppm\n")
    f.write(f"delta_CH4 = {delta:.2f} ppm\n")
    f.write(f"eksperyment = -2.3 ppm\n")

print("\nZapisano: CH4_opt.xyz, TMS_opt.xyz, wynik_NMR.txt")
print("Gotowe.")
