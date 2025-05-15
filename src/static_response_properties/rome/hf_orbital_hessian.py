import pdaggerq as rome
from pdaggerq.parser import contracted_strings_to_tensor_terms


def build_expression():
    pq = rome.pq_helper('fermi')
    ai = ['e1(a,i)']
    aihc = ['e1(i,a)']
    bj = ['e1(b,j)']
    bjhc = ['e1(j,b)']

    pq.add_double_commutator(1.0, ['f'], ai, bj)
    pq.add_double_commutator(-1.0, ['f'], ai, bjhc)
    pq.add_double_commutator(-1.0, ['f'], aihc, bj)
    pq.add_double_commutator(1.0, ['f'], aihc, bjhc)

    pq.add_double_commutator(1.0, ['v'], ai, bj)
    pq.add_double_commutator(-1.0, ['v'], ai, bjhc)
    pq.add_double_commutator(-1.0, ['v'], aihc, bj)
    pq.add_double_commutator(1.0, ['v'], aihc, bjhc)

    pq.simplify()
    return pq


def print_restricted(pq):
    print()
    terms = pq.strings()
    for term in terms:
        print(term)


def print_unrestricted(pq):
    print()
    for s1 in ['a', 'b']:
        for s2 in ['a', 'b']:
            spin_labels = {
                'j': s1,
                'b': s1,
                'i': s2,
                'a': s2,
            }
            terms = pq.strings(spin_labels=spin_labels)
            terms = contracted_strings_to_tensor_terms(terms)

            print(f'# The {s1}-{s2} spin-block:')
            for term in terms:
                print(f'# {term}')
                pyterm = term.einsum_string(
                    update_val='h_'+s1+s2,
                    output_variables=('j', 'b', 'i', 'a'),
                )
                print(f'{pyterm}')
            print()


def main():
    print('<HF| [[H, (aₐ†aᵢ-aᵢ†aₐ)], (aₐ†aᵢ-aᵢ†aₐ)] |HF>')
    pq = build_expression()
    use_spinorbitals = True
    if use_spinorbitals is False:
        print_restricted(pq)
    else:
        print_unrestricted(pq)


if __name__ == "__main__":
    main()
