def parse_tp_degrees(values: list[int]) -> list[int]:
    if not values:
        raise ValueError("至少要提供一个 TP 度数。")
    tp_degrees = [int(value) for value in values]
    if any(value <= 0 for value in tp_degrees):
        raise ValueError(f"TP 度数必须是正整数，当前拿到 {tp_degrees!r}。")
    return tp_degrees


def build_stage_rank_groups(tp_degrees: list[int]) -> list[list[int]]:
    stage_rank_groups = []
    rank_cursor = 0
    for tp_degree in tp_degrees:
        ranks = list(range(rank_cursor, rank_cursor + tp_degree))
        stage_rank_groups.append(ranks)
        rank_cursor += tp_degree
    return stage_rank_groups


def build_pp_rank_groups(stage_rank_groups: list[list[int]]) -> list[list[int]]:
    if not stage_rank_groups:
        return []
    if any(not ranks for ranks in stage_rank_groups):
        raise ValueError("stage_rank_groups 里不能出现空 stage。")

    max_tp_degree = max(len(ranks) for ranks in stage_rank_groups)
    pp_rank_groups = []
    for pp_idx in range(max_tp_degree):
        pp_rank_groups.append([ranks[min(pp_idx, len(ranks) - 1)] for ranks in stage_rank_groups])
    return pp_rank_groups


def build_p2p_lists(stage_rank_groups: list[list[int]], pp_rank_groups: list[list[int]] | None = None) -> dict:
    if pp_rank_groups is None:
        pp_rank_groups = build_pp_rank_groups(stage_rank_groups)

    world_size = sum(len(ranks) for ranks in stage_rank_groups)
    send_list = [[] for _ in range(world_size)]
    recv_list = [[] for _ in range(world_size)]
    send_empty_list = [[] for _ in range(world_size)]
    recv_empty_list = [[] for _ in range(world_size)]
    stage_leaders = [ranks[0] for ranks in stage_rank_groups]

    for pp_group in pp_rank_groups:
        for stage_idx, (src, dst) in enumerate(zip(pp_group, pp_group[1:])):
            is_empty = src != stage_leaders[stage_idx] or dst != stage_leaders[stage_idx + 1]
            send_list[src].append(dst)
            recv_list[dst].append(src)
            send_empty_list[src].append(is_empty)
            recv_empty_list[dst].append(is_empty)

    return {
        "send_list": send_list,
        "recv_list": recv_list,
        "send_empty_list": send_empty_list,
        "recv_empty_list": recv_empty_list,
    }


def build_hybrid_layout(tp_degrees: list[int]) -> dict:
    stage_rank_groups = build_stage_rank_groups(tp_degrees)
    pp_rank_groups = build_pp_rank_groups(stage_rank_groups)
    p2p_lists = build_p2p_lists(stage_rank_groups, pp_rank_groups)
    return {
        "tp_degrees": tp_degrees,
        "stage_rank_groups": stage_rank_groups,
        "pp_rank_groups": pp_rank_groups,
        "world_size": sum(tp_degrees),
        "num_stages": len(tp_degrees),
        **p2p_lists,
    }
