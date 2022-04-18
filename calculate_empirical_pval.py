import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def parse_separator(s):
    if s == 'tab':
        return '\t'
    elif s == 'comma':
        return ','
    else:
        raise ValueError(f'Unknown separator {s}')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate empirical p-value for a set of results")
    parser.add_argument('--observed', type=Path, required=True,
                        help='Path to observed results')
    parser.add_argument('--null_results', type=Path, required=True,
                        help='Path to null results', nargs='+')
    parser.add_argument('--group_by', type=str, required=True, nargs='+',
                        help='Variable(s) to group by')
    parser.add_argument('--out', type=Path, required=True,
                        help='Path to output file')
    parser.add_argument('--sep', type=parse_separator, default='tab', choices=['\t', ','],
                        help='Separator in input files')
    parser.add_argument('--score_column', type=str, default='score',
                        help='Column name for score')
    parser.add_argument('--comparison_operator', type=str, default='<', choices=['<', '>', '>=', '<='],
                        help='Comparison operator. Compares null_scores {operator} observed_score. If evals to true, it increases the p-value (becomes less significant)')

    return parser.parse_args()


def main(observed, null_results, group_by, out, sep, score_column, comparison_operator):
    observed = pd.read_csv(observed, index_col=0, sep=sep)
    null_results = [pd.read_csv(f, index_col=0, sep=sep) for f in null_results]
    n = len(null_results)

    results_df = pd.DataFrame()

    # Concatenate all dataframes
    null_results = pd.concat(null_results, axis=0)

    null_results['set'] = 'null'
    observed['set'] = 'observed'
    
    all_dfs = pd.concat([observed, null_results], axis=0)

    # Calculate empirical p-value
    try:
        grouper = all_dfs.groupby(group_by)
    except KeyError:
        raise ValueError(f'Group by variable(s) {group_by} not found in {observed}')

    for name, group in tqdm(grouper):
        if len(group['set'].unique()) != 2:
            continue

        observed_score = group[group['set'] == 'observed'][score_column].values[0]
        null_scores = group[group['set'] == 'null'][score_column].values
        print(name)
        print("Observed score:", observed_score)
        print("Null scores:", null_scores)
        # Calculate r 
        if comparison_operator == '<':
            null_scores = null_scores < observed_score
        elif comparison_operator == '>':
            null_scores = null_scores > observed_score
        elif comparison_operator == '>=':
            null_scores = null_scores >= observed_score
        elif comparison_operator == '<=':
            null_scores = null_scores <= observed_score
        else:
            raise ValueError("Unknown comparison operator")
        print(sum(null_scores),' + 1 / ', n, ' + 1' )
        pval = (sum(null_scores) + 1) / (n + 1)
        print("P-value:", pval) 

        # Print group if significant:
        if pval < 0.05:
            print("\n" + "-" * 80)
            print(f"Group: {name}")
            print(f"Observed score: {observed_score}")
            print(f"Null scores: {null_scores}")
            print(f"p-value: {pval}")
            print("-" * 80)

        # Add groupd result to results_df
        new_row = {'pval' : pval}
        new_row.update(dict(zip(group_by, name)))
        results_df = results_df.append(pd.DataFrame(new_row, index=[name]))

    # Write results to file
    results_df.to_csv(out, sep=sep)


if __name__ == '__main__':
    args = parse_args()
    main(args)
