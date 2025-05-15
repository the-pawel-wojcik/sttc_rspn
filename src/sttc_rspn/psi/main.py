from DePrinceLab.response.electronic_structure import scf
from DePrinceLab.response.intermediates_builders import extract_intermediates
from DePrinceLab.response.polarizabilities.core import pretty_print
from DePrinceLab.response.polarizabilities.operators import (
    OrbitalHessianAction,
)
from DePrinceLab.response.polarizabilities.solve_gmres import\
    calculate as pol_with_gmres


def main():

    print("Polarizabilities:")

    do_hf = False
    if do_hf:
        _, wfn = scf()
        intermediates = extract_intermediates(wfn)
        orbital_hessian_action = OrbitalHessianAction(intermediates)
        pol = pol_with_gmres(orbital_hessian_action, intermediates)
        print("3) from GMRES iterative solution:")
        pretty_print(pol)

    do_cc = True
    if do_cc:
        orbital_hessian_action = CCSDJacobian(intermediates)
        pol = pol_with_gmres(orbital_hessian_action, intermediates)
        print("4) from GMRES iterative solution (optimized build):")
        pretty_print(pol)


if __name__ == "__main__":
    main()
