import numpy as np
from basis_set import sto_1g, sto_3g
from basis_set import gto
from integration import get_fock_matrix, get_overlap_matrix, get_density_matrix


class Element():
    def __init__(self, element, basis_type, position):
        """

        :param element: the name of element
        :param basis_type: basis type sto-3g.. sto-1g..
        :param position: position of element
        """
        setattr(self, "name", element)
        setattr(self, "position", np.array(position))
        setattr(self, "basis_type", basis_type)
        if self.basis_type == "sto-1g":
            setattr(self, "basis_list", sto_1g[self.name])
        elif self.basis_type == "sto-3g":
            setattr(self, "basis_list", sto_3g[self.name])

        tmp_list = []
        for basis in self.basis_list:
            gto_list = []
            for gaussian in basis:
                gto_list.append(gto(*gaussian, self.position))
            tmp_list.append(gto_list)
        setattr(self, "basis_list", tmp_list)



class H(Element):
    def __init__(self, basis_type, position):
        super(H, self).__init__("H", basis_type, position)
        setattr(self, "charge", 1.0)


if __name__ == '__main__':
    pass
