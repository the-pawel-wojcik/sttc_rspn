import numpy as np
from numpy import einsum
from numpy.typing import NDArray
from DePrinceLab.response.intermediates_builders import Intermediates
from sttc_rspn.psi.ccsd.equations.doubles_uhf import (
    get_doubles_residual_aaaa,
    get_doubles_residual_abab,
    get_doubles_residual_abba,
    get_doubles_residual_baab,
    get_doubles_residual_baba,
    get_doubles_residual_bbbb,
)


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

    def find_t_amplitudes(self):
        MAX_CCSD_ITER = 50
        ENERGY_CONVERGENCE = 1e-6
        RESIDUALS_CONVERGENCE = 1e-6
        e_fmt = '12.6f'
        old_energy = self.get_energy()

        diis = {
            'aa': list(),
            'bb': list(),
            'aaaa': list(),
            'abab': list(),
            'abba': list(),
            'baab': list(),
            'baba': list(),
            'bbbb': list(),
        }

        for iter_idx in range(MAX_CCSD_ITER):
            current_t1_aa = np.zeros_like(self.t1_aa)

            singles_residual_aa = self.get_singles_residual_aa()
            singles_residual_bb = self.get_singles_residual_bb()

            diis['aa'].append(singles_residual_aa)
            diis['bb'].append(singles_residual_bb)

            args = (
                self.intermediates,
                self.t1_aa, self.t1_bb,
                self.t2_aaaa, self.t2_abab, self.t2_bbbb
            )
            diis['aaaa'].append(get_doubles_residual_aaaa(*args))
            diis['abab'].append(get_doubles_residual_abab(*args))
            diis['abba'].append(get_doubles_residual_abba(*args))
            diis['baab'].append(get_doubles_residual_baab(*args))
            diis['baba'].append(get_doubles_residual_baba(*args))
            diis['bbbb'].append(get_doubles_residual_bbbb(*args))

            residuals_norm = sum(
                np.linalg.norm(diis[residual][-1]) for residual in [
                    'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb'
                ]
            )

            print(f"Iteration {iter_idx:>2d}:", end='')
            current_energy = self.get_energy()
            energy_change = current_energy - old_energy
            print(f' {current_energy:{e_fmt}}', end='')
            print(f' {energy_change:{e_fmt}}')

            energy_converged = np.abs(energy_change) < ENERGY_CONVERGENCE
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE

            self.t1_aa = current_t1_aa
            if energy_converged and residuals_converged:
                break
            old_energy = current_energy
        else:
            raise RuntimeError("CCSD didn't converge")

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
        uhf_ccsd += 0.250 * einsum(
            'jiab,abji', g_aaaa[oa, oa, va, va], t2_aaaa
        )
        uhf_ccsd += 0.250 * einsum(
            'jiab,abji', g_abab[oa, ob, va, vb], t2_abab
        )
        uhf_ccsd += 0.250 * einsum(
            'ijab,abij', g_abab[oa, ob, va, vb], t2_abab
        )
        uhf_ccsd += 0.250 * einsum(
            'jiba,baji', g_abab[oa, ob, va, vb], t2_abab
        )
        uhf_ccsd += 0.250 * einsum(
            'ijba,baij', g_abab[oa, ob, va, vb], t2_abab
        )
        uhf_ccsd += 0.250 * einsum(
            'jiab,abji', g_bbbb[ob, ob, vb, vb], t2_bbbb
        )
        uhf_ccsd += -0.50 * einsum(
            'jiab,ai,bj', g_aaaa[oa, oa, va, va], t1_aa, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        uhf_ccsd += 0.50 * einsum(
            'ijab,ai,bj', g_abab[oa, ob, va, vb], t1_aa, t1_bb,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        uhf_ccsd += 0.50 * einsum(
            'jiba,ai,bj', g_abab[oa, ob, va, vb], t1_bb, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        uhf_ccsd += -0.50 * einsum(
            'jiab,ai,bj', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )

        return uhf_ccsd

    def get_singles_residual_aa(self) -> NDArray:
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_abab = self.g_abab
        g_bbbb = self.g_bbbb
        va = self.va
        vb = self.vb
        oa = self.oa
        ob = self.ob
        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_abab = self.t2_abab

        # The spin aa case.
        singles_res_aa = 1.00 * einsum('ai->ai', f_aa[va, oa])
        singles_res_aa += -1.00 * einsum('ji,aj->ai', f_aa[oa, oa], t1_aa)
        singles_res_aa += 1.00 * einsum('ab,bi->ai', f_aa[va, va], t1_aa)
        singles_res_aa += -1.00 * einsum('jb,baij->ai', f_aa[oa, va], t2_aaaa)
        singles_res_aa += 1.00 * einsum('jb,abij->ai', f_bb[ob, vb], t2_abab)
        singles_res_aa += -1.00 * einsum(
            'jb,aj,bi->ai', f_aa[oa, va], t1_aa, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += 1.00 * einsum(
            'jabi,bj->ai', g_aaaa[oa, va, va, oa], t1_aa
        )
        singles_res_aa += 1.00 * einsum(
            'ajib,bj->ai', g_abab[va, ob, oa, vb], t1_bb
        )
        singles_res_aa += -0.50 * einsum(
            'kjbi,bakj->ai', g_aaaa[oa, oa, va, oa], t2_aaaa
        )
        singles_res_aa += -0.50 * einsum(
            'kjib,abkj->ai', g_abab[oa, ob, oa, vb], t2_abab
        )
        singles_res_aa += -0.50 * einsum(
            'jkib,abjk->ai', g_abab[oa, ob, oa, vb], t2_abab
        )
        singles_res_aa += -0.50 * einsum(
            'jabc,bcij->ai', g_aaaa[oa, va, va, va], t2_aaaa
        )
        singles_res_aa += 0.50 * einsum(
            'ajbc,bcij->ai', g_abab[va, ob, va, vb], t2_abab
        )
        singles_res_aa += 0.50 * einsum(
            'ajcb,cbij->ai', g_abab[va, ob, va, vb], t2_abab
        )
        singles_res_aa += 1.00 * einsum(
            'kjbc,caik,bj->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_aa += -1.00 * einsum(
            'kjcb,caik,bj->ai', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_aa += 1.00 * einsum(
            'jkbc,acik,bj->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_aa += -1.00 * einsum(
            'kjbc,acik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_aa += 0.50 * einsum(
            'kjbc,cakj,bi->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += -0.50 * einsum(
            'kjbc,ackj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += -0.50 * einsum(
            'jkbc,acjk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += 0.50 * einsum(
            'kjbc,aj,bcik->ai', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += -0.50 * einsum(
            'jkbc,aj,bcik->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += -0.50 * einsum(
            'jkcb,aj,cbik->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += 1.00 * einsum(
            'kjbi,ak,bj->ai', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_aa += -1.00 * einsum(
            'kjib,ak,bj->ai', g_abab[oa, ob, oa, vb], t1_aa, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_aa += 1.00 * einsum(
            'jabc,bj,ci->ai', g_aaaa[oa, va, va, va], t1_aa, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += 1.00 * einsum(
            'ajcb,bj,ci->ai', g_abab[va, ob, va, vb], t1_bb, t1_aa,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_aa += 1.00 * einsum(
            'kjbc,ak,bj,ci->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 2), (0, 1)]
        )
        singles_res_aa += -1.00 * einsum(
            'kjcb,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 2), (0, 1)]
        )

        return singles_res_aa

    def get_singles_residual_bb(self) -> NDArray:
        f_aa = self.f_aa
        f_bb = self.f_bb
        g_aaaa = self.g_aaaa
        g_abab = self.g_abab
        g_bbbb = self.g_bbbb
        va = self.va
        vb = self.vb
        oa = self.oa
        ob = self.ob
        t1_aa = self.t1_aa
        t1_bb = self.t1_bb
        t2_aaaa = self.t2_aaaa
        t2_abab = self.t2_abab
        t2_bbbb = self.t2_bbbb
        singles_res_bb = 1.00 * einsum('ai->ai', f_bb[vb, ob])
        singles_res_bb += -1.00 * einsum('ji,aj->ai', f_bb[ob, ob], t1_bb)
        singles_res_bb += 1.00 * einsum('ab,bi->ai', f_bb[vb, vb], t1_bb)
        singles_res_bb += 1.00 * einsum('jb,baji->ai', f_aa[oa, va], t2_abab)
        singles_res_bb += -1.00 * einsum('jb,baij->ai', f_bb[ob, vb], t2_bbbb)
        singles_res_bb += -1.00 * einsum(
            'jb,aj,bi->ai', f_bb[ob, vb], t1_bb, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'jabi,bj->ai', g_abab[oa, vb, va, ob], t1_aa
        )
        singles_res_bb += 1.00 * einsum(
            'jabi,bj->ai', g_bbbb[ob, vb, vb, ob], t1_bb
        )
        singles_res_bb += -0.50 * einsum(
            'kjbi,bakj->ai', g_abab[oa, ob, va, ob], t2_abab
        )
        singles_res_bb += -0.50 * einsum(
            'jkbi,bajk->ai', g_abab[oa, ob, va, ob], t2_abab
        )
        singles_res_bb += -0.50 * einsum(
            'kjbi,bakj->ai', g_bbbb[ob, ob, vb, ob], t2_bbbb
        )
        singles_res_bb += 0.50 * einsum(
            'jabc,bcji->ai', g_abab[oa, vb, va, vb], t2_abab
        )
        singles_res_bb += 0.50 * einsum(
            'jacb,cbji->ai', g_abab[oa, vb, va, vb], t2_abab
        )
        singles_res_bb += -0.50 * einsum(
            'jabc,bcij->ai', g_bbbb[ob, vb, vb, vb], t2_bbbb
        )
        singles_res_bb += -1.00 * einsum(
            'kjbc,caki,bj->ai', g_aaaa[oa, oa, va, va], t2_abab, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'kjcb,caki,bj->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += -1.00 * einsum(
            'jkbc,caik,bj->ai', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'kjbc,caik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += -0.50 * einsum(
            'kjcb,cakj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += -0.50 * einsum(
            'jkcb,cajk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 0.50 * einsum(
            'kjbc,cakj,bi->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += -0.50 * einsum(
            'kjbc,aj,bcki->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += -0.50 * einsum(
            'kjcb,aj,cbki->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 0.50 * einsum(
            'kjbc,aj,bcik->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += -1.00 * einsum(
            'jkbi,ak,bj->ai', g_abab[oa, ob, va, ob], t1_bb, t1_aa,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'kjbi,ak,bj->ai', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb,
            optimize=['einsum_path', (0, 2), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'jabc,bj,ci->ai', g_abab[oa, vb, va, vb], t1_aa, t1_bb,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'jabc,bj,ci->ai', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb,
            optimize=['einsum_path', (0, 1), (0, 1)]
        )
        singles_res_bb += -1.00 * einsum(
            'jkbc,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb,
            optimize=['einsum_path', (0, 2), (1, 2), (0, 1)]
        )
        singles_res_bb += 1.00 * einsum(
            'kjbc,ak,bj,ci->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb,
            optimize=['einsum_path', (0, 2), (1, 2), (0, 1)]
        )
        return singles_res_bb
