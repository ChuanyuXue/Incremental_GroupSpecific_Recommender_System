from tools.tools import *
import numpy as np
import copy


def cold_update(counts: list, orig_groups: list, new_nodes: list)->list:
    origs_ = copy.deepcopy(orig_groups)
    cs = []
    for x in origs_:
        if len(x) > len(orig_groups):
            cs.append(np.mean(counts[x]))
        else:
            cs.append(1e5)
    if origs_:
        origs_[cs.index(min(cs))] += list(np.unique(new_nodes))
    else:
        origs_.append(list(np.unique(new_nodes)))
    return origs_


def uncold_update(similary: np.ndarray, origin_groups: list, new_nodes: list)->list:
    origs_ = copy.deepcopy(origin_groups)
    sp_hash = dict()
    sn_hash = dict()
    for i in range(len(similary)):
        ia = similary[i, :]
        sp_hash[i] = np.nansum(ia[ia > 0])
        sn_hash[i] = np.nansum(ia[ia < 0])
    sp = np.count_nonzero(similary > 0) + 1e-5
    sn = np.count_nonzero(similary < 0) + 1e-5

    for i in new_nodes:
        min_modurity = 1e9
        proper_group_index = -1
        origin_group_index = -1
        for g in origs_:
            summ = 0
            for k in g:
                summ += -1 * (similary[i][k] - sp_hash[i] * sp_hash[k] / sp - sn_hash[i]
                              * sn_hash[k] / sn)
            if summ < min_modurity:
                min_modurity = summ
                proper_group_index = origs_.index(g)
                origin_group_index = find_value_in_2dlist(i, origs_)
        if proper_group_index != -1:
            origs_[proper_group_index].append(i)
        if origin_group_index != -1:
            origs_[origin_group_index].remove(i)
    return origs_


def bipar_update(net: np.ndarray, origin_groups: Union[list, tuple], new_nodes: list)->list:
    net_ = copy.deepcopy(net)
    e = np.sum(net_ != 0)
    groups = copy.deepcopy(origin_groups)
    new_nodes = np.array(new_nodes.copy())
    for line_index in range(len(new_nodes)):
        u_dict, i_dict = group_to_dict(groups[0]), group_to_dict(groups[1])
        u, i, r = new_nodes[line_index][0], new_nodes[line_index][1], new_nodes[line_index][2]
        if u not in u_dict:
            continue
        else:
            net_[u][i] += r
            group_index = -1
            max_modurity = -1e9
            for i, value in enumerate(net_[u, :]):
                if value and i in i_dict:
                    modurity = net_[u][i] - (np.sum(net_[u, :]) * np.sum(net_[:, i])) / e
                    max_modurity = modurity if modurity > max_modurity else max_modurity
                    group_index = i_dict[i]
            if group_index != -1:
                groups[0][group_index].append(u)
                groups[0][u_dict[u]].remove(u)

        if i not in i_dict:
            continue
        else:
            net_[u][i] += r
            group_index = -1
            max_modurity = -1e9
            for u, value in enumerate(net_[:, i]):
                if value and u in u_dict:
                    modurity = net_[u][i] - (np.sum(net_[u, :]) * np.sum(net_[:, i])) / e
                    max_modurity = modurity if modurity > max_modurity else max_modurity
                    group_index = u_dict[u]
            if group_index != -1:
                groups[1][group_index].append(i)
                groups[1][i_dict[i]].remove(i)
    return groups
