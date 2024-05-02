import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", help="Path to the input file")
    parser.add_argument(
        "--output_file", help="Path to the output file", default="sampled_data.csv"
    )
    parser.add_argument(
        "--n_samples", help="Number of samples to take", type=int, default=1000
    )
    parser.add_argument(
        "--save2", help="save remaining data also?", action="store_true"
    )
    parser.add_argument(
        "--seed", help="Random seed to use for sampling", type=int, default=42
    )

    args = parser.parse_args()

    data = pd.read_csv(args.input_file)

    sampled_data = data.sample(n=args.n_samples, random_state=args.seed)

    sampled_data.to_csv(args.output_file, index=False)

    if args.save2:
        remaining_data = data.drop(sampled_data.index)
        remaining_data.to_csv("remaining_data.csv", index=False)


if __name__ == "__main__":
    main()

