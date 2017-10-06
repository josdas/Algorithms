from itertools import chain, combinations
from collections import defaultdict


def get_transaction_list(data):
    transaction_list = []
    items = set()
    for value in data:
        transaction_list.append(value)
        for element in value:
            items.add(frozenset([element]))
    return items, transaction_list


def get_items_with_min_support(items, transaction_list, min_support, frequencies=None):
    if frequencies is None:
        frequencies = defaultdict(int)
    for item in items:
        for transaction in transaction_list:
            if item in transaction:
                frequencies[item] += 1
    result_items = set(
        item
        for freq, item in frequencies.items()
        if freq >= min_support * len(transaction_list)
    )
    return result_items


def get_items_union_with_fixed_size(items, size):
    return set(
        x.union(y)
        for x in items
        for y in items
        if len(x.union(y)) == size
    )


def get_all_subset(_set):
    return chain(*(
        combinations(_set, i)
        for i in range(1, len(_set) + 1)
    ))


def run_apriori(data, min_support, min_confidence):
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
            confidence = get_support(item) / get_support(subset)
            if confidence >= min_confidence:
                rules.append(
                    (subset, item, confidence)
                )
    return rules
