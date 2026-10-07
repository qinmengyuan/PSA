import json
import argparse

def compute_reward(jsonl_path):
    total = 0
    scores = []

    for line in open(jsonl_path, "r", encoding="utf-8"):
        data = json.loads(line)
        total += 1

        final_score = data.get("final_score", 0)
        if not final_score:
            print(f"Warning: Missing final_score in data",data)
            input("Press Enter to continue...")

        scores.append(final_score)

    avg_score = sum(scores) / total if total > 0 else 0.0

    return {
        "total": total,
        "average_score": avg_score
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute attack success rate from jsonl file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to jsonl result file"
    )

    args = parser.parse_args()

    stats = compute_reward(args.input)

    print("====== Reward Statistics ======")
    print(f"Total samples        : {stats['total']}")  
    print(f"Average reward score : {stats['average_score']:.4f}")



if __name__ == "__main__":
    main()