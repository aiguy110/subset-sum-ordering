from itertools import product


def subset_sum_ordering(nums):
    """
    Given a list of positive integers, returns (partition, radices).

    partition: groups of the sorted input in ascending order. The last group
               (largest elements) is the most significant position in the
               mixed-radix scheme.
    radices:   2^(group size) for each group -- the digit alphabet size per position.

    Encoding a subset S as a digit tuple (d_0, ..., d_{k-1}), where d_i is the
    rank (by group-sum) of the elements of S drawn from group i, produces a
    total order over all subsets consistent with subset-sum ordering: if
    sum(S1) < sum(S2) then lex(S1) < lex(S2) (comparing right-to-left, i.e.
    the last group is most significant).

    Groups are cut greedily: a new group starts after element k when
        sum(sorted[0..k]) < sorted[k+1]
    This ensures no combination of elements from earlier groups can reach or
    exceed any nonempty selection from a later group, making the digit at each
    position unambiguously more significant than all positions to its left.
    This is a generalisation of the super-increasing property: a fully
    super-increasing sequence produces all singleton groups (pure binary encoding).
    """
    if not nums:
        return [], []

    sorted_nums = sorted(nums)
    groups = []
    current_group = []
    prefix_sum = 0

    for num in sorted_nums:
        if current_group and prefix_sum < num:
            groups.append(current_group)
            current_group = []
        current_group.append(num)
        prefix_sum += num

    groups.append(current_group)
    radices = [2 ** len(g) for g in groups]
    return groups, radices


def build_digit_map(group):
    """
    For a group of elements (sorted ascending), returns a list of length 2^n
    where index d gives (sum_contribution, inclusion_bitmask).

    Subsets are ranked by their sum (ties broken by bitmask value), so digit d
    is the d-th smallest subset by sum.  Bit i of the bitmask indicates whether
    group[i] is included.
    """
    n = len(group)
    return sorted(
        (sum(group[i] for i in range(n) if mask >> i & 1), mask)
        for mask in range(2 ** n)
    )


def verify(partition, radices):
    """
    Verify that the (partition, radices) encoding is sum-order-consistent.

    Iterates every digit tuple using "sweep" ordering: for each inclusion
    pattern (each group either excluded=digit 0, or included=digit>0),
    enumerated in MSB-first lex order, then within each pattern sweeps all
    non-zero digit values for included groups with the LSB group varying
    fastest.  When a multi-element group is first included it is incremented
    all the way through its non-zero values before the enclosing positions
    advance.

    Returns (True, None) on success, or (False, diagnostic_dict) on the first
    violation found.
    """
    k = len(partition)
    if k == 0:
        return True, None

    digit_maps = [build_digit_map(g) for g in partition]

    prev_sum = None
    i = 0
    # Enumerate inclusion patterns: each bit says whether group g is included
    # (digit > 0).  Patterns are in MSB-first lex order so the largest group
    # varies slowest, matching the overall mixed-radix significance ordering.
    for pattern_bits in product(*([range(2)] * k)):
        # pattern_bits[j] is the inclusion bit for group (k-1-j); reverse to
        # index by group number.
        bits = pattern_bits[::-1]  # bits[g] = 1 if group g is included

        # Build the range of digit values for each group in MSB-first order so
        # the inner product keeps the LSB group varying fastest.
        ranges = []
        for g in range(k - 1, -1, -1):
            ranges.append(range(1, radices[g]) if bits[g] else range(1))

        for combo_msb_first in product(*ranges):
            digits = combo_msb_first[::-1]  # align: digits[g] <-> partition[g]
            total = sum(digit_maps[g][digits[g]][0] for g in range(k))

            if prev_sum is not None and total < prev_sum:
                return False, {
                    "at_index": i,
                    "digits": digits,
                    "sum": total,
                    "prev_sum": prev_sum,
                }
            prev_sum = total
            i += 1

    return True, None


if __name__ == "__main__":
    examples = [
        [1, 20, 5, 6, 2],   # motivating example
        [1, 2, 3, 4],       # prefix sum touches next element: [1],[2,3,4]
        [3, 5, 6, 7],       # no element dominates prefix
        [1, 1, 2, 4],       # multiset: equal elements force grouping
        [1, 2, 4, 8, 16],   # super-increasing: all singletons, pure binary
        [10],               # single element
    ]

    for nums in examples:
        partition, radices = subset_sum_ordering(nums)
        valid, diag = verify(partition, radices)
        total_subsets = 1
        for r in radices:
            total_subsets *= r
        print(f"Input:          {nums}")
        print(f"Partition:      {partition}")
        print(f"Radices:        {radices}  ({total_subsets} subsets)")
        print(f"Valid:          {valid}" + (f"  FAIL: {diag}" if not valid else ""))
        print()
