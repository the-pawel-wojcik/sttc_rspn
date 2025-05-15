import numpy as np
from numpy import einsum
from DePrinceLab.response.intermediates_builders import Intermediates


class UHF_CCSD:
    def __init__(self, intermediates: Intermediates) -> None:
        noa = intermediates.noa
        nva = intermediates.nmo - noa
        nob = intermediates.nob
        nvb = intermediates.nmo - nob
        self.f_aa = intermediates.f_aa
        self.f_bb = intermediates.f_bb
        self.g_aaaa = intermediates.g_aaaa
        self.g_abab = intermediates.g_abab
        self.g_bbbb = intermediates.g_bbbb
        self.oa = intermediates.oa
        self.ob = intermediates.ob
        self.va = intermediates.va
        self.vb = intermediates.vb

        self.t1_aa = np.zeros(shape=(nva, noa))
        self.t1_bb = np.zeros(shape=(nvb, nob))

        self.t2_aaaa = np.zeros(shape=(nva, nva, noa, noa))
        self.t2_abab = np.zeros(shape=(nva, nvb, noa, nob))
        self.t2_bbbb = np.zeros(shape=(nvb, nvb, nob, nob))

    def get_energy(self) -> float:
        f_aa = self.f_aa
        f_bb = self.f_bb
        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_abab = self.t2_abab
        t2_bbbb = self.t2_bbbb
        g_aaaa = self.g_aaaa
        g_abab = self.g_abab
        g_bbbb = self.g_bbbb
        oa = self.oa
        ob = self.ob
        va = self.va
        vb = self.vb

        uhf_ccsd = 0.0
        uhf_ccsd += 1.00 * einsum('ii', f_aa[oa, oa])
        uhf_ccsd += 1.00 * einsum('ii', f_bb[ob, ob])
        uhf_ccsd += 1.00 * einsum('ia,ai', f_aa[oa, va], t1_aa)
        uhf_ccsd += 1.00 * einsum('ia,ai', f_bb[ob, vb], t1_bb)
        uhf_ccsd += -0.50 * einsum('jiji', g_aaaa[oa, oa, oa, oa])
        uhf_ccsd += -0.50 * einsum('jiji', g_abab[oa, ob, oa, ob])
        uhf_ccsd += -0.50 * einsum('ijij', g_abab[oa, ob, oa, ob])
        uhf_ccsd += -0.50 * einsum('jiji', g_bbbb[ob, ob, ob, ob])
        uhf_ccsd += 0.250 * einsum('jiab,abji', g_aaaa[oa, oa, va, va], t2_aaaa)
        uhf_ccsd += 0.250 * einsum('jiab,abji', g_abab[oa, ob, va, vb], t2_abab)
        uhf_ccsd += 0.250 * einsum('ijab,abij', g_abab[oa, ob, va, vb], t2_abab)
        uhf_ccsd += 0.250 * einsum('jiba,baji', g_abab[oa, ob, va, vb], t2_abab)
        uhf_ccsd += 0.250 * einsum('ijba,baij', g_abab[oa, ob, va, vb], t2_abab)
        uhf_ccsd += 0.250 * einsum('jiab,abji', g_bbbb[ob, ob, vb, vb], t2_bbbb)
        uhf_ccsd += -0.50 * einsum('jiab,ai,bj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        uhf_ccsd += 0.50 * einsum('ijab,ai,bj', g_abab[oa, ob, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
        uhf_ccsd += 0.50 * einsum('jiba,ai,bj', g_abab[oa, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
        uhf_ccsd += -0.50 * einsum('jiab,ai,bj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

        return uhf_ccsd
