import os
import json
import time
import numpy as np

# 1. Wczytanie konfiguracji z zewnętrznego pliku
with open('config.json', 'r') as f:
    config = json.load(f)

# Pobranie zamrożonej bazy odniesienia TMS i innych gotowych cząsteczek
zablokowane_bazy = config.get('zablokowane_bazy_odniesienia', {})
sigma_tms = zablokowane_bazy.get('TMS', None)

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

def run_nmr(mol_xyz, name, target_symmetry, freeze_atoms=None, charge=0, spin=0):
    t0 = time.time()
    print(f"\n{'='*50}")
    print(f"START: {name}")
    print(f"{'='*50}")

    # 1. Inicjalizacja molekuły (zawsze bez symetrii na starcie dla geomeTRIC)
    mol = gto.Mole()
    if isinstance(mol_xyz, list):
        mol.atom = "\n".join(mol_xyz)
    else:
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
    mf.canonical_orthog_thresh = 0 
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF nie zbiegło dla {name}")

    print(f"\n[{name}] Optymalizacja geometrii...")
    if freeze_atoms:
        print(f"Wykryto atomy do zamrożenia: {freeze_atoms}. Uruchamiam relaksację wodorów...")
        constraints_file = "constraints.txt"
        with open(constraints_file, "w") as cf:
            cf.write("$freeze\n")
            for idx in freeze_atoms:
                cf.write(f"xyz {idx + 1}\n")
        
        # geomeTRIC modyfikuje obiekt mol bezpośrednio w miejscu (in-place)
        geometric_solver.optimize(mf, maxsteps=100, convergence='tight', constraints=constraints_file)
        if os.path.exists(constraints_file):
            os.remove(constraints_file)
    else:
        print("Brak zablokowanych atomów. Wykonuję pełną optymalizację.")
        geometric_solver.optimize(mf, maxsteps=100, convergence='tight')

    mol.tofile(f'{name}_opt.xyz')

    # 2. Drugie SCF wykonujemy bezpośrednio na tym samym obiekcie 'mol', który ma już zaktualizowane współrzędne 3D
    print(f"\n[{name}] SCF na zoptymalizowanej geometrii (docelowy grid)...")
    
    # Jeśli użytkownik jawnie wymusił grupę inną niż C1/auto/false, przypisujemy ją tutaj
    if target_symmetry and str(target_symmetry).lower() not in ["auto", "false", "c1"]:
        mol.symmetry = target_symmetry
        mol.build(0, 0) # Przebudowanie tabel symetrii bez resetowania współrzędnych
    else:
        mol.symmetry = False

    mf_eq = dft.RKS(mol)
    mf_eq.max_memory = MEMORY
    mf_eq.xc = XC
    mf_eq.grids.level = GRID_LEVEL
    mf_eq.conv_tol = 1e-10
    mf_eq.kernel()
    if not mf_eq.converged:
        raise RuntimeError(f"SCF po optymalizacji nie zbiegło dla {name}")

    # 3. Obliczenia GIAO-NMR
    print(f"\n[{name}] Liczenie GIAO-NMR...")
    nmr_obj = nmr.rks.NMR(mf_eq)

    # Wyciąganie wyników
    shielding_tensors = nmr_obj.kernel()

    c_indices = [i for i, a in enumerate(mol.elements) if a == 'C']
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

# Słownik na bieżące wyniki
nowe_wyniki = {}
# Parsujemy sekcję "czasteczki_do_obliczen" zamiast dawnej "molecules"
for name, data in config.get('czasteczki_do_obliczen', {}).items():
    freeze = data.get('freeze_atoms', None)
    symmetry = data.get('symmetry', 'auto') # domyślnie auto, jeśli brak w JSON
    
    sigma_c = run_nmr(data['xyz'], name, symmetry, freeze_atoms=freeze)
    
    nowe_wyniki[name] = {
        'sigma_c_all': sigma_c,
        'sigma_c_avg': np.mean(sigma_c),
        'sigma_c_std': np.std(sigma_c)
    }

# Generowanie raportu końcowego łączącego stare i nowe dane
print(f"\n{'='*50}")
print("PODSUMOWANIE PRZESUNIĘĆ CHEMICZNYCH (13C)")
print(f"{'='*50}")
print(f"Baza odniesienia (TMS): σ_iso = {sigma_tms:.3f} ppm\n")

with open('wynik_NMR.txt', 'w') as f:
    f.write(f"Metoda: {XC}/{BASIS}\n")
    f.write(f"Wzorzec TMS: sigma = {sigma_tms:.3f} ppm\n\n")
    
    # Najpierw wypisujemy zablokowane z pliku konfiguracyjnego
    for name, sigma_avg in zablokowane_bazy.items():
        if name == 'TMS':
            continue
        delta = sigma_tms - sigma_avg
        output_line = f" Czarsteczka: {name:<8} | σ_iso = {sigma_avg:.3f} ppm | δ 13C = {delta:.2f} ppm (Wczytano z bazy)"
        print(output_line)
        f.write(output_line + "\n")
        
    # Następnie dopisujemy świeżo obliczone struktury (np. Toluen)
    for name, res in nowe_wyniki.items():
        delta = sigma_tms - res['sigma_c_avg']
        output_line = f" Czarsteczka: {name:<8} | σ_iso = {res['sigma_c_avg']:.3f} ppm | δ 13C = {delta:.2f} ppm"
        print(output_line)
        f.write(output_line + "\n")

print("\nZapisano pliki struktur zoptymalizowanych oraz zbiorczy wynik_NMR.txt.")
print("Gotowe.")
