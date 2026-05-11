#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from statistics import mean, stdev


def update_xml_for_run(root, seed, max_seconds):
    experiment = root.find("framework").find("experiment")
    experiment.attrib["random_seed"] = str(int(seed))

    loop_functions = root.find("loop_functions")
    settings = loop_functions.find("settings")

    # This is the CPFA loop-function time limit used in your XML.
    settings.attrib["MaxSimTimeInSeconds"] = str(int(max_seconds))

    # Keep framework length large enough so ARGoS does not stop earlier.
    # Your current XML uses length="6000", so this usually does not need to change.
    current_length = int(float(experiment.attrib.get("length", "6000")))
    if current_length < max_seconds:
        experiment.attrib["length"] = str(int(max_seconds))


def run_single_task(task):
    xml_path, seed, minute, tmp_dir = task
    max_seconds = minute * 60

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        update_xml_for_run(root, seed, max_seconds)

        with tempfile.NamedTemporaryFile(
            "wb",
            suffix=".argos",
            prefix=f"eval_seed_{seed}_m{minute}_",
            dir=tmp_dir,
            delete=False,
        ) as tmpf:
            tree.write(tmpf, encoding="utf-8", xml_declaration=True)
            tmp_path = tmpf.name

        try:
            result = subprocess.run(
                ["argos3", "-n", "-c", tmp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if result.returncode != 0:
                return {
                    "seed": seed,
                    "minute": minute,
                    "status": "failed",
                    "cumulative_fitness": None,
                    "raw_last_line": "",
                    "stderr": result.stderr.strip(),
                }

            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if not lines:
                return {
                    "seed": seed,
                    "minute": minute,
                    "status": "no_output",
                    "cumulative_fitness": None,
                    "raw_last_line": "",
                    "stderr": result.stderr.strip(),
                }

            last_line = lines[-1]
            cumulative_fitness = float(last_line.split(",")[0])

            return {
                "seed": seed,
                "minute": minute,
                "status": "ok",
                "cumulative_fitness": cumulative_fitness,
                "raw_last_line": last_line,
                "stderr": result.stderr.strip(),
            }

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return {
            "seed": seed,
            "minute": minute,
            "status": "exception",
            "cumulative_fitness": None,
            "raw_last_line": "",
            "stderr": str(e),
        }


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed CPFA parameters at each minute using repeated ARGoS runs."
    )
    parser.add_argument("-x", "--xml", required=True, help="Fixed XML file with selected best parameters")
    parser.add_argument("-n", "--runs", type=int, default=50, help="Number of random seeds")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Parallel workers")
    parser.add_argument("-o", "--output-prefix", default="random", help="Output file prefix")
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--minutes", type=int, default=12)
    parser.add_argument("--tmp-dir", default=None)

    args = parser.parse_args()

    xml_path = os.path.abspath(args.xml)
    tmp_dir = args.tmp_dir or os.path.dirname(xml_path) or "."
    os.makedirs(tmp_dir, exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.runs)]

    tasks = []
    for seed in seeds:
        for minute in range(1, args.minutes + 1):
            tasks.append((xml_path, seed, minute, tmp_dir))

    print("=" * 70)
    print("Per-minute evaluation by repeated duration runs")
    print(f"XML: {xml_path}")
    print(f"Seeds: {args.runs}")
    print(f"Minutes: {args.minutes}")
    print(f"Total ARGoS runs: {len(tasks)}")
    print(f"Workers: {args.jobs}")
    print("=" * 70)

    results = []

    with Pool(args.jobs) as pool:
        for idx, row in enumerate(pool.imap_unordered(run_single_task, tasks), start=1):
            print(
                f"[{idx}/{len(tasks)}] "
                f"seed={row['seed']} minute={row['minute']} "
                f"status={row['status']} cumulative={row['cumulative_fitness']}"
            )
            results.append(row)

    results.sort(key=lambda r: (r["seed",] if False else r["seed"], r["minute"]))

    raw_csv = args.output_prefix + "_cumulative_by_duration_raw.csv"
    cumulative_csv = args.output_prefix + "_cumulative_by_run.csv"
    per_minute_csv = args.output_prefix + "_per_minute_by_run.csv"
    summary_csv = args.output_prefix + "_per_minute_summary.csv"

    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "minute", "status", "cumulative_fitness", "raw_last_line", "stderr"],
        )
        writer.writeheader()
        writer.writerows(results)

    # Build lookup: seed -> minute -> cumulative score
    by_seed = {seed: {} for seed in seeds}
    for row in results:
        if row["status"] == "ok" and row["cumulative_fitness"] is not None:
            by_seed[row["seed"]][row["minute"]] = row["cumulative_fitness"]

    valid_seeds = [
        seed for seed in seeds
        if all(m in by_seed[seed] for m in range(1, args.minutes + 1))
    ]

    print()
    print(f"Valid seeds with all {args.minutes} minute checkpoints: {len(valid_seeds)}/{args.runs}")

    # Cumulative by run
    with open(cumulative_csv, "w", newline="") as f:
        fieldnames = ["seed"] + [f"minute_{m}" for m in range(1, args.minutes + 1)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in valid_seeds:
            row = {"seed": seed}
            for m in range(1, args.minutes + 1):
                row[f"minute_{m}"] = by_seed[seed][m]
            writer.writerow(row)

    # Per-minute increments by run
    per_minute_by_seed = {}

    with open(per_minute_csv, "w", newline="") as f:
        fieldnames = ["seed"] + [f"minute_{m}" for m in range(1, args.minutes + 1)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for seed in valid_seeds:
            previous = 0.0
            increments = []
            row = {"seed": seed}

            for m in range(1, args.minutes + 1):
                cumulative = by_seed[seed][m]
                increment = cumulative - previous
                previous = cumulative

                increments.append(increment)
                row[f"minute_{m}"] = increment

            per_minute_by_seed[seed] = increments
            writer.writerow(row)

    # Summary: mean/std per minute across seeds
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["minute", "mean_collected", "std_collected", "n_runs"],
        )
        writer.writeheader()

        for minute_idx in range(args.minutes):
            values = [per_minute_by_seed[seed][minute_idx] for seed in valid_seeds]
            writer.writerow({
                "minute": minute_idx + 1,
                "mean_collected": mean(values) if values else "",
                "std_collected": safe_std(values) if values else "",
                "n_runs": len(values),
            })

    print("=" * 70)
    print(f"Saved raw cumulative results: {raw_csv}")
    print(f"Saved cumulative by run:     {cumulative_csv}")
    print(f"Saved per-minute by run:     {per_minute_csv}")
    print(f"Saved per-minute summary:    {summary_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()