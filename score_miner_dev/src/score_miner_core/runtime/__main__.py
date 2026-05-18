from score_miner_core.runtime.memory_budget import MemoryBudget


def main() -> None:
    print(MemoryBudget().status())


if __name__ == "__main__":
    main()
