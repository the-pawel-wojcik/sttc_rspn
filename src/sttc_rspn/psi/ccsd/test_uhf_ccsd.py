from DePrinceLab.response.electronic_structure import scf
from DePrinceLab.response.intermediates_builders import extract_intermediates
from sttc_rspn.psi.ccsd.find_cluster_amplitudes import UHF_CCSD


mol, scf_energy, wfn = scf()
intermediates = extract_intermediates(wfn)
ccsd = UHF_CCSD(intermediates)

assert ccsd.get_energy() + mol.nuclear_repulsion_energy() == scf_energy
