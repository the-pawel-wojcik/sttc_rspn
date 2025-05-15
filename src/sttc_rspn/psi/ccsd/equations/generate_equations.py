import itertools
import pdaggerq


def numpy_print_energy_uhf(pq):
    print('import numpy as np')
    print('from numpy import einsum')
    print()
    from pdaggerq.parser import contracted_strings_to_tensor_terms

    out_var = 'uhf_ccsd'
    print(f'{out_var} = np.zeros()  # TODO')

    terms = pq.strings(spin_labels={})
    tensor_terms = contracted_strings_to_tensor_terms(terms)

    for my_term in tensor_terms:
        einsum_terms = my_term.einsum_string(
            output_variables=(),
            update_val=out_var,
        )
        print(f"{einsum_terms}")


def numpy_print_singles_uhf(pq):
    print('import numpy as np')
    print('from numpy import einsum')
    print()
    from pdaggerq.parser import contracted_strings_to_tensor_terms
    for spin_mix in itertools.product(['a', 'b'], repeat=2):
        spin_labels = {
            'a': spin_mix[0],
            'i': spin_mix[1],
        }

        out_var = 'singles_res_' + ''.join(spin_mix)
        print(f"# The spin {"".join(spin_mix)} case.")
        print(f'{out_var} = np.zeros()  # TODO')

        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)

        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'i'),
                update_val=out_var,
            )
            print(f"{einsum_terms}")


def numpy_print_doubles_uhf(pq):
    print('import numpy as np')
    print('from numpy import einsum')
    print()
    from pdaggerq.parser import contracted_strings_to_tensor_terms
    for spin_mix in itertools.product(['a', 'b'], repeat=4):
        out_var = 'doubles_res_' + ''.join(spin_mix)
        print(f"# The spin {"".join(spin_mix)} case.")
        print(f'{out_var} = np.zeros()  # TODO')
        spin_labels = {
            'a': spin_mix[0],
            'b': spin_mix[1],
            'i': spin_mix[2],
            'j': spin_mix[3],
        }
        terms = pq.strings(spin_labels=spin_labels)
        tensor_terms = contracted_strings_to_tensor_terms(terms)

        for my_term in tensor_terms:
            einsum_terms = my_term.einsum_string(
                output_variables=('a', 'b', 'i', 'j'),
                update_val=out_var
            )
            print(f"{einsum_terms}")


def build_energy():
    pq = pdaggerq.pq_helper('fermi')
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def build_singles():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['a*(i)', 'a(a)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def build_doubles():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators([['a*(i)', 'a*(j)', 'a(a)', 'a(b)']])
    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])
    pq.simplify()
    return pq


def main():
    do_energy = True
    do_singles = False
    do_doubles = False

    if do_energy is True:
        pq = build_energy()
        numpy_print_energy_uhf(pq)

    if do_singles is True:
        pq = build_singles()
        numpy_print_singles_uhf(pq)

    elif do_doubles is True:
        pq = build_doubles()
        numpy_print_doubles_uhf(pq)


if __name__ == "__main__":
    main()
