import os
import json
import time
import numpy as np

# 1. Wczytanie konfiguracji z zewnętrznego pliku
with open('config.json', 'r') as f:
    config = json.load(f)

settings = config['settings']

# 2. Ustawienia środowiskowe i inicjalizacja pamięci u źródła
os.environ["OMP_NUM_THREADS"] = str(settings['threads'])

import pyscf
pyscf.lib.param.MAX_MEMORY = settings['memory_mb']

from pyscf import gto, dft
from pyscf.prop import nmr
from pyscf.geomopt import geometric_solver

# Poprawki dla stabilności NumPy w PySCF
import pyscf.dft.numint
import pyscf.sgx.sgx_jk
pyscf.dft.numint.BLKSIZE = 192
pyscf.sgx.sgx_jk.BLKSIZE = 192

XC = settings['xc']
BASIS = settings['basis']
VERBOSE = settings['verbose']
GRID_LEVEL = settings['grid_level']
MEMORY = settings['memory_mb']

def run_nmr(mol_xyz, name, target_symmetry, charge=0, spin=0):
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"START: {name}")
    print(f"{'='*50}")

    # Krok 1: Optymalizacja geometrii (bez jawnej symetrii, aby solver geomeTRIC przeszedł bez błędów)
    mol = gto.Mole()
    mol.atom = mol_xyz
    mol.basis = BASIS
    mol.unit = 'Angstrom'
    mol.charge = charge
    mol.spin = spin
    mol.verbose = VERBOSE
    mol.symmetry = False
    mol.max_memory = MEMORY
    mol.build()
    print(f"Atomy: {mol.natm}, elektrony: {mol.tot_electrons()}")

    print(f"\n[{name}] SCF startowe...")
    mf = dft.RKS(mol)
    mf.max_memory = MEMORY
    mf.xc = XC
    mf.grids.level = 2
    mf.conv_tol = 1e-9
    mf.canonical_orthog_thresh = 0 # dodane
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF nie zbiegło dla {name}")

    print(f"\n[{name}] Optymalizacja geometrii...")
    mol_eq = geometric_solver.optimize(mf, maxsteps=100, convergence='tight')
    mol_eq.tofile(f'{name}_opt.xyz')

    print(f"\n[{name}] SCF na optymalnej geometrii...")
    coords_eq = mol_eq.atom_coords()
    
    # Krok 2: Rekonstrukcja molekuły i wymuszenie symetrii dla dokładnego NMR
    mol_final = gto.Mole()
    mol_final.atom = [[mol_eq.atom_symbol(i), coords_eq[i]] for i in range(mol_eq.natm)]
    mol_final.basis = BASIS
    mol_final.unit = 'Bohr'
    mol_final.charge = charge
    mol_final.spin = spin
    mol_final.verbose = VERBOSE
    #mol_final.symmetry = target_symmetry
    mol_final.symmetry = True if target_symmetry == "auto" else target_symmetry
    mol_final.max_memory = MEMORY
    mol_final.build()

    mf_eq = dft.RKS(mol_final)
    mf_eq.max_memory = MEMORY
    mf_eq.xc = XC
    mf_eq.grids.level = GRID_LEVEL
    mf_eq.conv_tol = 1e-10
    mf_eq.kernel()
    if not mf_eq.converged:
        raise RuntimeError(f"SCF po opt nie zbiegło dla {name}")

    # Krok 3: Obliczenia GIAO-NMR
    print(f"\n[{name}] Liczenie GIAO-NMR...")
    nmr_obj = nmr.rks.NMR(mf_eq)
    nmr_obj.kernel()

    shielding_tensors = nmr_obj.shielding()

    # Filtrowanie indeksów atomów węgla przy użyciu bezpiecznej właściwości .elements
    c_indices = [i for i, a in enumerate(mol_final.elements) if a == 'C']
    sigma_c = []
    for idx in c_indices:
        tensor = shielding_tensors[idx]
        sigma_iso = np.trace(tensor) / 3.0
        sigma_c.append(sigma_iso)
        print(f"[{name}] C{idx}: σ_iso = {sigma_iso:.3f} ppm")

    t1 = time.time()
    print(f"[{name}] Gotowe w {t1-t0:.1f} s")
    return np.array(sigma_c)

# Główna pętla wykonawcza
print(f"Metoda: {XC}/{BASIS}")
print("Uruchamiam sekwencję obliczeniową z pliku konfiguracyjnego...")

results = {}
for name, data in config['molecules'].items():
    sigma_c = run_nmr(data['xyz'], name, data['symmetry'])
    results[name] = {
        'sigma_c_all': sigma_c,
        'sigma_c_avg': np.mean(sigma_c),
        'sigma_c_std': np.std(sigma_c)
    }

# Generowanie raportu końcowego
print(f"\n{'='*50}")
print("PODSUMOWANIE PRZESUNIĘĆ CHEMICZNYCH (13C)")
print(f"{'='*50}")

sigma_tms = results['TMS']['sigma_c_avg']
print(f"Baza odniesienia (TMS): σ_iso = {sigma_tms:.3f} ppm (Rozrzut: {results['TMS']['sigma_c_std']:.3f} ppm)\n")

with open('wynik_NMR.txt', 'w') as f:
    f.write(f"Metoda: {XC}/{BASIS}\n")
    f.write(f"Wzorzec TMS: sigma = {sigma_tms:.3f} ppm\n\n")
    
    for name, res in results.items():
        if name == 'TMS':
            continue
        delta = sigma_tms - res['sigma_c_avg']
        output_line = f" Czarsteczka: {name:<8} | σ_iso = {res['sigma_c_avg']:.3f} ppm | δ 13C = {delta:.2f} ppm"
        print(output_line)
        f.write(output_line + "\n")

print("\nZapisano pliki struktur zoptymalizowanych oraz zbiorczy wynik_NMR.txt.")
print("Gotowe.")
