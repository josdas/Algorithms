from itertools import chain, combinations
from collections import defaultdict


def get_transaction_list(data):
    """
    :param data: iterator
    :return: it returns (items, transaction_list)
    """
    transaction_list = []
    items = set()
    for value in data:
        transaction_list.append(value)
        for element in value:
            items.add(frozenset([element]))
    return items, transaction_list


def get_items_with_min_support(items, transaction_list, min_support, frequencies=None):
    """
    :param items: set of sets
    :param transaction_list: array of sequences
    :param min_support:
    :param frequencies: dictionary which contains sets. If it is not None it will be updated during function
    :return: it returns all sets with needed support level
    """
    if frequencies is None:
        frequencies = defaultdict(int)
    for item in items:
        for transaction in transaction_list:
            if item.issubset(transaction):
                frequencies[item] += 1
    result_items = set(
        item
        for item in items
        if frequencies[item] > min_support * len(transaction_list)
    )
    return result_items


def get_items_union_with_fixed_size(items, size):
    """
    :param items: set of sets
    :param size: needed size
    :return: it returns set of all unions of the needed size
    """
    return set(
        x.union(y)
        for x in items
        for y in items
        if len(x.union(y)) == size
    )


def get_all_subset(_set):
    """
    :param _set: set
    :return: it returns iterator on frozenset of all subsets of _set
    """
    return map(frozenset,
               chain(*(
                   combinations(_set, i)
                   for i in range(1, len(_set) + 1)
               )))


def run_apriori(data, min_support, min_confidence):
    """
    :param data: iterator
    :return: it returns list of rules. Each of the rules is (set, prediction, probability)
    """
    frequencies = defaultdict(int)
    items, transaction_list = get_transaction_list(data)
    cur_set = get_items_with_min_support(items,
                                         transaction_list,
                                         min_support,
                                         frequencies)
    cur_level = 1
    all_set = []
    while cur_set:
        all_set.extend(list(cur_set))
        next_set = get_items_union_with_fixed_size(cur_set,
                                                   cur_level + 1)
        next_set = get_items_with_min_support(next_set,
                                              transaction_list,
                                              min_support,
                                              frequencies)
        cur_set = next_set
        cur_level += 1

    def get_support(_item):
        return float(frequencies[_item]) / len(transaction_list)

    rules = []

    for item in all_set:
        for subset in get_all_subset(item):
            if subset and subset != item:
                confidence = get_support(item) / get_support(frozenset(subset))
                if confidence > min_confidence:
                    rules.append(
                        (tuple(subset), tuple(item), confidence)
                    )
    return rules


if __name__ == '__main__':
    _data = [[1, 2, 3], [2, 3], [2], [1, 3], [2, 3, 4]]
    _min_support = 0.0
    _min_confidence = 0
    result = run_apriori(_data, _min_support, _min_confidence)
    print(result)
