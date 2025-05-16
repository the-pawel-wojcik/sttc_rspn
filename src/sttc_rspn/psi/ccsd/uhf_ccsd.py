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


class DIIS:

    def __init__(self, noa: int, nva: int, nob: int, nvb: int) -> None:
        self.diis_coefficients = None
        self.storage_size = 0
        self.noa = noa
        self.nva = nva
        self.nob = nob
        self.nvb = nvb
        total_residual_dim = nva * noa + nvb * nob
        total_residual_dim += nva * nva * noa * noa
        total_residual_dim += nva * nvb * noa * nob
        total_residual_dim += nvb * nvb * nob * nob
        self.residuals_matrix = np.zeros(shape=(total_residual_dim, 0))
        self.start_idx = 3

    def find_new_vec(self, guess, change):
        if self.storage_size < self.start_idx:
            return guess

        self.add_new_residual(guess)
        # if len > max_lex self.residuals_matrix.pop(0)
        matrix, rhs = self.build_linear_problem()
        c = np.linalg.solve(matrix, rhs)
        assert self.residuals_matrix.shape[1] == c.shape[0]
        new_guess = self.residuals_matrix @ c
        return new_guess

    def add_new_residual(self, residuals: dict[str, NDArray]) -> None:
        """ Hoping that the new residual looks like this
        ```
        residuals = {
            'aa': NDarray,
            'bb': NDarray,
            'aaaa': NDarray,
            'abab': NDarray,
            'abba': NDarray,
            'baab': NDarray,
            'baba': NDarray,
            'bbbb': NDarray,
        }
        ```
        """
        noa = self.noa
        nob = self.nob
        nva = self.nva
        nvb = self.nvb
        self.residuals_matrix = np.hstack(
            (
                self.residuals_matrix,
                np.hstack(
                    (
                        residuals['aa'].reshape(nva * noa),
                        residuals['bb'].reshape(nvb * nob),
                        residuals['aaaa'].reshape(nva * nva * noa * noa),
                        residuals['abab'].reshape(nva * nvb * noa * nob),
                        residuals['bbbb'].reshape(nvb * nvb * nob * nob),
                    )
                ).reshape(-1, 1),
            )
        )
        self.storage_size += 1

    def build_linear_problem(self) -> tuple[NDArray, NDArray]:
        dim = len(self.residuals_matrix[-1])
        matrix = self.residuals_matrix @ self.residuals_matrix.T
        matrix = np.vstack((
            matrix,
            -1 * np.ones((1, dim))
        ))
        matrix = np.hstack((
            matrix, -1 * np.ones((dim + 1, 1))
        ))
        matrix[-1][-1] = 0.0
        rhs = np.zeros((dim + 1, 1))
        rhs[-1][0] = -1
        return matrix, rhs


class UHF_CCSD:

    def __init__(self, intermediates: Intermediates) -> None:
        self.intermediates = intermediates
        noa = intermediates.noa
        nva = intermediates.nmo - noa
        nob = intermediates.nob
        nvb = intermediates.nmo - nob
        self.f_aa = intermediates.f_aa
        self.f_bb = intermediates.f_bb
        self.g_aaaa = intermediates.g_aaaa
        self.g_abab = intermediates.g_abab
        self.g_bbbb = intermediates.g_bbbb
        self.noa = noa
        self.nva = nva
        self.nob = nob
        self.nvb = nvb
        self.oa = intermediates.oa
        self.ob = intermediates.ob
        self.va = intermediates.va
        self.vb = intermediates.vb

        self.t1_aa = np.zeros(shape=(nva, noa))
        self.t1_bb = np.zeros(shape=(nvb, nob))

        self.t2_aaaa = np.zeros(shape=(nva, nva, noa, noa))
        self.t2_abab = np.zeros(shape=(nva, nvb, noa, nob))
        self.t2_bbbb = np.zeros(shape=(nvb, nvb, nob, nob))

        oa = self.intermediates.oa
        va = self.intermediates.va
        ob = self.intermediates.ob
        vb = self.intermediates.vb
        new_axis = np.newaxis

        fock_energy_a = self.intermediates.f_aa.diagonal()
        fock_energy_b = self.intermediates.f_bb.diagonal()

        # a set of matrices where for each matrix the index [a][i] or
        # [a][b][i][j] (you get the point) gives you the inverse of the sum
        # of the fock eigenvalues for these indices e.g
        # dampers['aa'][a][i] = 1 / (-fock_aa[a][a] + fock_aa[i][i])
        # See that the values are attempted to be negative bc, the virtual
        # eigenvalues come with a minus sign.
        self.dampers = {
            'aa': 1.0 / (
                - fock_energy_a[va, new_axis]
                + fock_energy_a[new_axis, oa]
            ),
            'bb': 1.0 / (
                - fock_energy_b[vb, new_axis]
                + fock_energy_b[new_axis, ob]
            ),
            'aaaa': 1.0 / (
                - fock_energy_a[va, new_axis, new_axis, new_axis]
                - fock_energy_a[new_axis, va, new_axis, new_axis]
                + fock_energy_a[new_axis, new_axis, oa, new_axis]
                + fock_energy_a[new_axis, new_axis, new_axis, oa]
            ),
            'abab': 1.0 / (
                - fock_energy_a[va, new_axis, new_axis, new_axis]
                - fock_energy_b[new_axis, vb, new_axis, new_axis]
                + fock_energy_a[new_axis, new_axis, oa, new_axis]
                + fock_energy_b[new_axis, new_axis, new_axis, ob]
            ),
            'bbbb': 1.0 / (
                - fock_energy_b[vb, new_axis, new_axis, new_axis]
                - fock_energy_b[new_axis, vb, new_axis, new_axis]
                + fock_energy_b[new_axis, new_axis, ob, new_axis]
                + fock_energy_b[new_axis, new_axis, new_axis, ob]
            ),
        }

    def solve_cc_equations(self):
        MAX_CCSD_ITER = 50
        ENERGY_CONVERGENCE = 1e-6
        RESIDUALS_CONVERGENCE = 1e-6

        # diis = DIIS(noa=self.noa, nva=self.nva, nob=self.nob, nvb=self.nvb)

        for iter_idx in range(MAX_CCSD_ITER):
            residuals = self.calculate_residuals()
            new_t_amps = self.calculate_new_amplitudes(residuals)

            old_energy = self.get_energy()
            self.update_t_amps(new_t_amps)
            current_energy = self.get_energy()
            energy_change = current_energy - old_energy
            residuals_norm = self.get_residuals_norm(residuals)
            self.print_iteration_report(
                iter_idx, current_energy, energy_change, residuals_norm
            )

            energy_converged = np.abs(energy_change) < ENERGY_CONVERGENCE
            residuals_converged = residuals_norm < RESIDUALS_CONVERGENCE

            if energy_converged and residuals_converged:
                break
            old_energy = current_energy
        else:
            raise RuntimeError("CCSD didn't converge")

    def get_residuals_norm(self, residuals):
        return sum(
            np.linalg.norm(residuals[component]) for component in [
                'aa', 'bb', 'aaaa', 'abab', 'abba', 'baab', 'baba', 'bbbb'
            ]
        )

    def print_iteration_report(
        self, iter_idx, current_energy, energy_change, residuals_norm,
    ):
        e_fmt = '12.6f'
        print(f"Iteration {iter_idx:>2d}:", end='')
        print(f' {current_energy:{e_fmt}}', end='')
        print(f' {energy_change:{e_fmt}}', end='')
        print(f' {residuals_norm:{e_fmt}}')

    def update_t_amps(self, new_t_amps):
        self.t1_aa = new_t_amps['aa']
        self.t1_bb = new_t_amps['bb']
        self.t2_aaaa = new_t_amps['aaaa']
        self.t2_abab = new_t_amps['abab']
        self.t2_bbbb = new_t_amps['bbbb']

    def calculate_residuals(self):
        residuals = dict()

        residuals['aa'] = self.get_singles_residual_aa()
        residuals['bb'] = self.get_singles_residual_bb()

        kwargs = dict(
            intermediates=self.intermediates,
            t1_aa=self.t1_aa,
            t1_bb=self.t1_bb,
            t2_aaaa=self.t2_aaaa,
            t2_abab=self.t2_abab,
            t2_bbbb=self.t2_bbbb,
        )
        residuals['aaaa'] = get_doubles_residual_aaaa(**kwargs)
        residuals['abab'] = get_doubles_residual_abab(**kwargs)
        residuals['abba'] = get_doubles_residual_abba(**kwargs)
        residuals['baab'] = get_doubles_residual_baab(**kwargs)
        residuals['baba'] = get_doubles_residual_baba(**kwargs)
        residuals['bbbb'] = get_doubles_residual_bbbb(**kwargs)

        return residuals

    def calculate_new_amplitudes(self, residuals):
        new_t_amps = dict()
        new_t_amps['aa'] =\
            self.t1_aa + residuals['aa'] * self.dampers['aa']
        new_t_amps['bb'] =\
            self.t1_bb + residuals['bb'] * self.dampers['bb']
        new_t_amps['aaaa'] =\
            self.t2_aaaa + residuals['aaaa'] * self.dampers['aaaa']
        new_t_amps['abab'] =\
            self.t2_abab + residuals['abab'] * self.dampers['abab']
        new_t_amps['bbbb'] =\
            self.t2_bbbb + residuals['bbbb'] * self.dampers['bbbb']

        return new_t_amps

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
