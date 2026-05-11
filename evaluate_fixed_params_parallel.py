#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from multiprocessing import Pool


def run_single_task(task):
    xml_path, seed, tmp_dir = task

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        experiment = root.find("framework").find("experiment")
        experiment.attrib["random_seed"] = str(int(seed))

        with tempfile.NamedTemporaryFile(
            "wb",
            suffix=".argos",
            prefix=f"eval_seed_{seed}_",
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
                    "fitness": None,
                    "status": "failed",
                    "raw_last_line": "",
                    "stderr": result.stderr.strip(),
                }

            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]

            if not lines:
                return {
                    "seed": seed,
                    "fitness": None,
                    "status": "no_output",
                    "raw_last_line": "",
                    "stderr": result.stderr.strip(),
                }

            last_line = lines[-1]
            fitness = float(last_line.split(",")[0])

            return {
                "seed": seed,
                "fitness": fitness,
                "status": "ok",
                "raw_last_line": last_line,
                "stderr": result.stderr.strip(),
            }

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return {
            "seed": seed,
            "fitness": None,
            "status": "exception",
            "raw_last_line": "",
            "stderr": str(e),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parallel evaluation for one fixed CPFA parameter set."
    )
    parser.add_argument("-x", "--xml", required=True, help="Path to fixed XML/.argos file")
    parser.add_argument("-n", "--runs", type=int, default=50, help="Number of evaluation runs")
    parser.add_argument("-o", "--output", default="evaluation_results.csv", help="Output CSV file")
    parser.add_argument("--seed-start", type=int, default=1000, help="First random seed")
    parser.add_argument("-j", "--jobs", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--tmp-dir", default=None, help="Temporary directory for generated ARGoS files")

    args = parser.parse_args()

    xml_path = os.path.abspath(args.xml)
    tmp_dir = args.tmp_dir or os.path.dirname(xml_path) or "."
    os.makedirs(tmp_dir, exist_ok=True)

    seeds = [args.seed_start + i for i in range(args.runs)]
    tasks = [(xml_path, seed, tmp_dir) for seed in seeds]

    print("=" * 60)
    print("Fixed-Parameter Parallel Evaluation")
    print(f"XML file: {xml_path}")
    print(f"Runs: {args.runs}")
    print(f"Parallel workers: {args.jobs}")
    print(f"Seed range: {seeds[0]} to {seeds[-1]}")
    print("=" * 60)

    results = []

    with Pool(args.jobs) as pool:
        for idx, row in enumerate(pool.imap_unordered(run_single_task, tasks), start=1):
            print(f"[{idx}/{args.runs}] seed={row['seed']} status={row['status']} fitness={row['fitness']}")
            results.append(row)

    results.sort(key=lambda r: r["seed"])

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["seed", "fitness", "status", "raw_last_line", "stderr"],
        )
        writer.writeheader()
        writer.writerows(results)

    valid = [r["fitness"] for r in results if r["status"] == "ok" and r["fitness"] is not None]

    print()
    print("=" * 60)
    print(f"Saved results to: {args.output}")
    print(f"Successful runs: {len(valid)}/{args.runs}")

    if valid:
        mean_val = sum(valid) / len(valid)
        print(f"Average fitness: {mean_val:.2f}")
        print(f"Min fitness: {min(valid):.2f}")
        print(f"Max fitness: {max(valid):.2f}")

    failed = [r for r in results if r["status"] != "ok"]
    if failed:
        print()
        print("Failed runs:")
        for r in failed:
            print(f"  seed={r['seed']} status={r['status']} stderr={r['stderr'][:200]}")

    print("=" * 60)


if __name__ == "__main__":
    main()