from DePrinceLab.response.electronic_structure import scf
from DePrinceLab.response.intermediates_builders import extract_intermediates
from sttc_rspn.psi.ccsd.uhf_ccsd import UHF_CCSD


mol, scf_energy, wfn = scf()
intermediates = extract_intermediates(wfn)
ccsd = UHF_CCSD(intermediates)
ccsd.solve_cc_equations()

nuclear_repulsion_energy = mol.nuclear_repulsion_energy()
print(f'Final UHF CCSD energy = {ccsd.get_energy() + nuclear_repulsion_energy:.6f}')
# assert ccsd.get_energy() + mol.nuclear_repulsion_energy() == scf_energy
