"""
CountsOutlierDetector — multidimensional categorical/binned-numeric outlier detection
based on the rarity of value combinations across columns.

This is a packaged, slightly cleaned version of the original detector. The public
API (``CountsOutlierDetector.fit_predict``) is unchanged.
"""

import os
import math
import random
import itertools
import concurrent.futures
from datetime import datetime, timedelta
from statistics import mean

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CountsOutlierDetector:
    def __init__(self,
                 n_bins=7,
                 bin_names=None,
                 max_dimensions=3,
                 threshold=0.05,
                 check_marginal_probs=False,
                 max_num_combinations=100_000,
                 min_values_per_column=2,
                 max_values_per_column=25,
                 results_folder="",
                 results_name="",
                 run_parallel=False,
                 verbose=False):
        if n_bins < 2:
            n_bins = 2
        if max_dimensions > 6:
            max_dimensions = 6

        self.n_bins = n_bins
        self.max_dimensions = max_dimensions
        self.threshold = threshold
        self.check_marginal_probs = check_marginal_probs
        self.max_num_combinations = max_num_combinations
        self.min_values_per_column = min_values_per_column
        self.max_values_per_column = max_values_per_column
        self.results_folder = results_folder
        self.results_name = results_name
        self.run_parallel = run_parallel
        self.verbose = verbose

        if bin_names is not None:
            if len(bin_names) != n_bins:
                bin_names = None
        if bin_names is not None:
            self.bin_names = bin_names
        else:
            defaults = {
                2: ['Low', 'High'],
                3: ['Low', 'Med', 'High'],
                4: ['Low', 'Med-Low', 'Med-High', 'High'],
                5: ['Low', 'Med-Low', 'Med', 'Med-High', 'High'],
                6: ['Very Low', 'Low', 'Med-Low', 'Med-High', 'High', 'Very High'],
                7: ['Very Low', 'Low', 'Med-Low', 'Med', 'Med-High', 'High', 'Very High'],
            }
            self.bin_names = defaults.get(n_bins, [f'Bin {x}' for x in range(n_bins)])

        self.col_types_arr = []
        self.numeric_col_names = None
        self.ordinal_encoders_arr = []
        self.orig_df = None
        self.data_df = None
        self.data_np = None
        self.bin_edges = None
        self.flagged_rows_df = None
        self.run_summary = None
        self.dimensions_examined = None
        self.rare_1d_values = None
        self.rare_2d_values = None
        self.unique_vals = None
        self.num_unique_vals = None

        np.random.seed(0)
        random.seed(0)

    def fit_predict(self, input_data):
        def update_run_summary(arr):
            for uc in sorted(set(arr)):
                self.run_summary += f"\nNumber of rows given score: {uc:2}: {arr.count(uc):5}"

        def get_unique_vals():
            uv_arr, num_uv = [], []
            for i in range(num_cols):
                uv = sorted(self.data_df.iloc[:, i].unique())
                uv_arr.append(uv)
                num_uv.append(len(uv))
            return uv_arr, num_uv

        def get_2d_fractions(i, j):
            two_d_fractions, two_d_row_nums = [], []
            for i_val in self.unique_vals[i]:
                i_vals_fractions, i_vals_row_nums = [], []
                cond1 = (self.data_np[:, i] == i_val)
                for j_val in self.unique_vals[j]:
                    cond2 = (self.data_np[:, j] == j_val)
                    rows_both = np.where(cond1 & cond2)
                    i_vals_fractions.append(len(rows_both[0]) / num_rows)
                    i_vals_row_nums.append(rows_both[0])
                two_d_fractions.append(i_vals_fractions)
                two_d_row_nums.append(i_vals_row_nums)
            return two_d_fractions, two_d_row_nums

        def get_1d_stats():
            fractions_1d = [[]] * num_cols
            rare_1d_values = [[]] * num_cols
            outliers_1d_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows

            for i in range(num_cols):
                col_threshold = (1.0 / self.num_unique_vals[i]) * self.threshold
                col_fractions_1d, col_rare_1d_values = [], []
                for v in self.unique_vals[i]:
                    frac = self.data_df.iloc[:, i].tolist().count(v) / num_rows
                    col_fractions_1d.append(frac)
                    rare_values_flag = (frac < col_threshold) and (frac < 0.01)
                    if rare_values_flag:
                        for r in np.where(self.data_np[:, i] == v)[0]:
                            outliers_1d_arr[r] += 1
                            expl = [[self.data_df.columns[i]], [self._get_col_value(i, v)]]
                            if outliers_explanation_arr[r] == []:
                                outliers_explanation_arr[r] = [expl]
                            else:
                                outliers_explanation_arr[r].append(expl)
                    col_rare_1d_values.append(rare_values_flag)
                fractions_1d[i] = col_fractions_1d
                rare_1d_values[i] = col_rare_1d_values

            self.run_summary += (f"\n\n1d: Number of common values (over all columns): "
                                 f"{flatten(rare_1d_values).count(False)}")
            self.run_summary += f"\n1d: Number of rare values: {flatten(rare_1d_values).count(True)}"
            update_run_summary(outliers_1d_arr)
            return fractions_1d, rare_1d_values, outliers_1d_arr, outliers_explanation_arr

        def get_2d_stats():
            fractions_2d, rare_2d_values = [], []
            outliers_2d_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows
            for _ in range(num_cols):
                fractions_2d.append([[]] * num_cols)
                rare_2d_values.append([[]] * num_cols)

            for i in range(num_cols - 1):
                for j in range(i + 1, num_cols):
                    local_fractions, two_d_row_nums = get_2d_fractions(i, j)
                    fractions_2d[i][j] = local_fractions
                    i_rare_arr = []
                    expected_under_uniform = 1.0 / (len(self.unique_vals[i]) * len(self.unique_vals[j]))
                    for i_vals_idx in range(len(fractions_2d[i][j])):
                        j_rare_arr = []
                        for j_vals_idx in range(len(fractions_2d[i][j][i_vals_idx])):
                            current_fraction = fractions_2d[i][j][i_vals_idx][j_vals_idx]
                            if self.check_marginal_probs:
                                expected_given_marginal = (
                                    fractions_1d[i][i_vals_idx]
                                    * fractions_1d[j][j_vals_idx]
                                    * self.threshold
                                )
                            else:
                                expected_given_marginal = np.inf
                            rare_value_flag = ((not rare_1d_values[i][i_vals_idx]) and
                                               (not rare_1d_values[j][j_vals_idx]) and
                                               (current_fraction < expected_under_uniform * self.threshold) and
                                               (current_fraction < expected_given_marginal) and
                                               (current_fraction < 0.01))
                            if rare_value_flag:
                                for r in two_d_row_nums[i_vals_idx][j_vals_idx]:
                                    outliers_2d_arr[r] += 1
                                    expl = [[self.data_df.columns[i], self.data_df.columns[j]],
                                            [self._get_col_value(i, i_vals_idx),
                                             self._get_col_value(j, j_vals_idx)]]
                                    if not outliers_explanation_arr[r]:
                                        outliers_explanation_arr[r] = [expl]
                                    else:
                                        outliers_explanation_arr[r].append(expl)
                            j_rare_arr.append(rare_value_flag)
                        i_rare_arr.append(j_rare_arr)
                    rare_2d_values[i][j] = i_rare_arr

            out = flatten(rare_2d_values)
            self.run_summary += f"\n\n2d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n2d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_2d_arr)
            return fractions_2d, rare_2d_values, outliers_2d_arr, outliers_explanation_arr

        def _run_higher_dim(dim, inner_fn, *extra_rare):
            fractions_xd = [[]] * num_cols
            rare_xd_values = [[]] * num_cols
            outliers_xd_arr = [0] * num_rows
            outliers_explanation_arr = [[]] * num_rows

            run_parallel = self.run_parallel and (
                self.__get_num_combinations(dim=dim) >= 1_000_000
            )

            if run_parallel:
                process_arr = []
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    for i in range(num_cols):
                        f = executor.submit(
                            inner_fn,
                            self, i, self.data_np, num_cols, num_rows,
                            self.unique_vals, fractions_1d, rare_1d_values, rare_2d_values,
                            *extra_rare, self.threshold,
                        )
                        process_arr.append(f)
                    for f_idx, f in enumerate(process_arr):
                        rare_arr_for_i, outliers_for_i, expl_for_i = f.result()
                        rare_xd_values[f_idx] = rare_arr_for_i
                        outliers_xd_arr = [x + y for x, y in zip(outliers_xd_arr, outliers_for_i)]
                        outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, expl_for_i)]
            else:
                for i in range(num_cols):
                    rare_arr_for_i, outliers_for_i, expl_for_i = inner_fn(
                        self, i, self.data_np, num_cols, num_rows,
                        self.unique_vals, fractions_1d, rare_1d_values, rare_2d_values,
                        *extra_rare, self.threshold,
                    )
                    rare_xd_values[i] = rare_arr_for_i
                    outliers_xd_arr = [x + y for x, y in zip(outliers_xd_arr, outliers_for_i)]
                    outliers_explanation_arr = [x + y for x, y in zip(outliers_explanation_arr, expl_for_i)]

            out = flatten(rare_xd_values)
            self.run_summary += f"\n\n{dim}d: Number of common combinations (over all columns): {out.count(False):,}"
            self.run_summary += f"\n{dim}d: Number of rare combinations: {out.count(True)}"
            update_run_summary(outliers_xd_arr)
            return fractions_xd, rare_xd_values, outliers_xd_arr, outliers_explanation_arr

        def create_output_csv(*arrs):
            (o1, o2, o3, o4, o5, o6, e1, e2, e3, e4, e5, e6) = arrs
            df = pd.DataFrame({
                "1d Counts": o1, "2d Counts": o2, "3d Counts": o3,
                "4d Counts": o4, "5d Counts": o5, "6d Counts": o6,
                "1d Explanations": e1, "2d Explanations": e2, "3d Explanations": e3,
                "4d Explanations": e4, "5d Explanations": e5, "6d Explanations": e6,
            })
            for d in range(1, 7):
                df[f'Any at {d}d'] = df[f'{d}d Counts'] > 0
            df['Any up to 1d'] = df['1d Counts'] > 0
            for d in range(2, 7):
                df[f'Any up to {d}d'] = df[f'Any up to {d-1}d'] | (df[f'{d}d Counts'] > 0)
            df['Any Scored'] = sum(df[f'{d}d Counts'] for d in range(1, 7)) > 0

            if self.results_folder:
                os.makedirs(self.results_folder, exist_ok=True)
                dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
                file_name = os.path.join(
                    self.results_folder,
                    f"{self.results_name}_results_{dt_string}.csv",
                )
                df.to_csv(file_name)
            return df

        def create_return_dict():
            return {
                'Scores': self.flagged_rows_df['TOTAL SCORE'],
                'Breakdown All Rows': self.flagged_rows_df,
                'Breakdown Flagged Rows': self.__output_explanations(),
                'Flagged Summary': run_summary_df,
            }

        self.orig_df = pd.DataFrame(input_data).copy().reset_index(drop=True)
        self.data_df = pd.DataFrame(input_data).copy().reset_index(drop=True)
        self.orig_df.columns = [str(x) for x in self.orig_df.columns]
        self.data_df.columns = [str(x) for x in self.data_df.columns]

        self.col_types_arr = self.__get_col_types_arr()
        self.numeric_col_names = [self.data_df.columns[x]
                                  for x in range(len(self.col_types_arr)) if self.col_types_arr[x] == 'N']

        drop_col_names_arr = []
        for c in range(len(self.data_df.columns)):
            if self.col_types_arr[c] == 'C':
                nun = self.data_df[self.data_df.columns[c]].nunique()
                if nun < self.min_values_per_column or nun > self.max_values_per_column:
                    drop_col_names_arr.append(self.data_df.columns[c])
        self.data_df = self.data_df.drop(columns=drop_col_names_arr)

        self.col_types_arr = self.__get_col_types_arr()
        self.numeric_col_names = [self.data_df.columns[x]
                                  for x in range(len(self.col_types_arr)) if self.col_types_arr[x] == 'N']

        for col_idx, col_name in enumerate(self.data_df.columns):
            if self.col_types_arr[col_idx] == 'C':
                mode_val = self.data_df[col_name].mode()
                if len(mode_val) > 0:
                    self.data_df[col_name] = self.data_df[col_name].fillna(mode_val.iloc[0])
            else:
                self.data_df[col_name] = self.data_df[col_name].fillna(self.data_df[col_name].median())

        for col_idx, col_name in enumerate(self.data_df.columns):
            if self.col_types_arr[col_idx] == 'N':
                self.data_df[col_name] = self.data_df[col_name].replace(-np.inf, self.data_df[col_name].min())
                self.data_df[col_name] = self.data_df[col_name].replace(np.inf, self.data_df[col_name].max())

        est = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')
        if len(self.numeric_col_names):
            xt = est.fit_transform(self.data_df[self.numeric_col_names])
            self.bin_edges = est.bin_edges_
            for col_idx, col_name in enumerate(self.numeric_col_names):
                self.data_df[col_name] = xt[:, col_idx].astype(int)

        num_cols = len(self.data_df.columns)
        num_rows = len(self.data_df)

        self.run_summary = f"\nNumber of rows: {num_rows}\nNumber of columns: {num_cols}"

        run_summary_df = pd.DataFrame(columns=[
            'Checked_2d', 'Checked_3d', 'Checked_4d', 'Checked_5d', 'Checked_6d',
            'Percent Flagged as 1d', 'Percent Flagged as 2d', 'Percent Flagged as 3d',
            'Percent Flagged as 4d', 'Percent Flagged as 5d', 'Percent Flagged as 6d',
            'Percent Flagged up to 1d', 'Percent Flagged up to 2d', 'Percent Flagged up to 3d',
            'Percent Flagged up to 4d', 'Percent Flagged up to 5d', 'Percent Flagged up to 6d',
            'Percent Flagged',
        ])

        if num_cols < 2:
            self.run_summary += (
                "\nLess than two columns found (after dropping columns with too few or too many unique values). "
                "Cannot determine outliers."
            )
            empty_arr = [0] * num_rows
            self.flagged_rows_df = create_output_csv(
                empty_arr, empty_arr, empty_arr, empty_arr, empty_arr, empty_arr,
                [""] * num_rows, [""] * num_rows, [""] * num_rows,
                [""] * num_rows, [""] * num_rows, [""] * num_rows,
            )
            self.flagged_rows_df['TOTAL SCORE'] = 0
            self.dimensions_examined = 0
            return create_return_dict()

        self.__ordinal_encode()
        self.data_np = self.data_df.values
        self.unique_vals, self.num_unique_vals = get_unique_vals()
        self.run_summary += f"\nCardinality of the columns (after binning numeric columns): {self.num_unique_vals}"

        fractions_1d, rare_1d_values, outliers_1d_arr, explanations_1d_arr = get_1d_stats()
        self.rare_1d_values = rare_1d_values
        self.dimensions_examined = 1

        checked = {d: False for d in range(2, 7)}
        outliers = {d: [0] * num_rows for d in range(2, 7)}
        explanations = {d: [""] * num_rows for d in range(2, 7)}
        rare_2d_values = rare_3d_values = rare_4d_values = rare_5d_values = None

        if self.max_dimensions >= 2:
            nc = self.__get_num_combinations(dim=2)
            if nc <= self.max_num_combinations:
                _, rare_2d_values, outliers[2], explanations[2] = get_2d_stats()
                checked[2] = True
                self.rare_2d_values = rare_2d_values
                self.dimensions_examined = 2

        if self.max_dimensions >= 3 and rare_2d_values is not None:
            nc = self.__get_num_combinations(dim=3)
            if num_cols >= 3 and nc <= self.max_num_combinations:
                _, rare_3d_values, outliers[3], explanations[3] = _run_higher_dim(
                    3, process_inner_loop_3d
                )
                checked[3] = True
                self.dimensions_examined = 3

        if self.max_dimensions >= 4 and rare_3d_values is not None:
            nc = self.__get_num_combinations(dim=4)
            if num_cols >= 4 and nc <= self.max_num_combinations:
                _, rare_4d_values, outliers[4], explanations[4] = _run_higher_dim(
                    4, process_inner_loop_4d, rare_3d_values
                )
                checked[4] = True
                self.dimensions_examined = 4

        if self.max_dimensions >= 5 and rare_4d_values is not None:
            nc = self.__get_num_combinations(dim=5)
            if num_cols >= 5 and nc <= self.max_num_combinations:
                _, rare_5d_values, outliers[5], explanations[5] = _run_higher_dim(
                    5, process_inner_loop_5d, rare_3d_values, rare_4d_values
                )
                checked[5] = True
                self.dimensions_examined = 5

        if self.max_dimensions >= 6 and rare_5d_values is not None:
            nc = self.__get_num_combinations(dim=6)
            if num_cols >= 6 and nc <= self.max_num_combinations:
                _, _, outliers[6], explanations[6] = _run_higher_dim(
                    6, process_inner_loop_6d, rare_3d_values, rare_4d_values, rare_5d_values
                )
                checked[6] = True
                self.dimensions_examined = 6

        self.flagged_rows_df = create_output_csv(
            outliers_1d_arr, outliers[2], outliers[3], outliers[4], outliers[5], outliers[6],
            explanations_1d_arr, explanations[2], explanations[3],
            explanations[4], explanations[5], explanations[6],
        )

        col_names = []
        for dim in range(self.max_dimensions + 1):
            cn = f'{dim}d Counts'
            if cn in self.flagged_rows_df.columns:
                col_names.append(cn)
        self.flagged_rows_df['TOTAL SCORE'] = self.flagged_rows_df[col_names].sum(axis=1)

        self.run_summary += "\n"
        for dim in range(1, self.max_dimensions + 1):
            n_scored = list(self.flagged_rows_df[f'Any at {dim}d'] > 0).count(True)
            self.run_summary += (
                f"\nNumber of rows flagged as outliers examining {dim}d: {n_scored:3}"
                f" ({round(n_scored*100.0/num_rows, 3)}%)"
            )

        new_df = pd.DataFrame(np.array([[
            checked[2], checked[3], checked[4], checked[5], checked[6],
            self.flagged_rows_df['Any at 1d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 2d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 3d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 4d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 5d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any at 6d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 1d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 2d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 3d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 4d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 5d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any up to 6d'].sum() * 100.0 / num_rows,
            self.flagged_rows_df['Any Scored'].sum() * 100.0 / num_rows,
        ]]), columns=run_summary_df.columns)
        run_summary_df = pd.concat([run_summary_df, new_df])
        for d in range(2, 7):
            run_summary_df[f'Checked_{d}d'] = run_summary_df[f'Checked_{d}d'].astype(bool)

        return create_return_dict()

    def get_most_flagged_rows(self):
        if self.flagged_rows_df is None or self.flagged_rows_df.empty:
            return pd.DataFrame()
        idx = (self.flagged_rows_df[self.flagged_rows_df['TOTAL SCORE'] > 0]
               .copy().sort_values('TOTAL SCORE', ascending=False))
        ret = self.orig_df.loc[idx.index].copy()
        ret.insert(0, 'TOTAL SCORE', idx['TOTAL SCORE'])
        return ret

    def __output_explanations(self):
        if self.flagged_rows_df is None or self.flagged_rows_df.empty:
            return None
        df_subset = self.flagged_rows_df[self.flagged_rows_df['Any Scored']].copy()
        expl_arr = []
        index_arr = list(df_subset.index)
        dims = self.dimensions_examined or 1
        for i in range(len(df_subset)):
            row = df_subset.iloc[i]
            row_expl = [index_arr[i]] + [""] * dims
            for j in range(1, dims + 1):
                row_expl[j] = row[f"{j}d Explanations"]
            expl_arr.append(row_expl)
        cols = (['Row Index']
                + [f'{j}d Explanations' for j in range(1, dims + 1)])
        return pd.DataFrame(expl_arr, columns=cols)

    def __get_col_types_arr(self):
        col_types_arr = ['N'] * len(self.data_df.columns)
        for col_idx, col_name in enumerate(self.data_df.columns):
            num_unique = self.data_df[col_name].nunique()
            if not self.__get_is_numeric(col_name):
                col_types_arr[col_idx] = 'C'
            if num_unique <= (2 * self.n_bins):
                col_types_arr[col_idx] = 'C'
        for col_idx, col_name in enumerate(self.data_df.columns):
            if col_types_arr[col_idx] == 'N':
                if self.data_df[col_name].dtype not in [int, np.int64]:
                    self.data_df[col_name] = self.data_df[col_name].astype(float)
        return col_types_arr

    def __ordinal_encode(self):
        self.ordinal_encoders_arr = [None] * len(self.data_df.columns)
        for col_idx, col_name in enumerate(self.data_df.columns):
            if self.col_types_arr[col_idx] == 'C':
                enc = OrdinalEncoder()
                self.ordinal_encoders_arr[col_idx] = enc
                x_np = enc.fit_transform(
                    np.asarray(self.orig_df[col_name].astype(str)).reshape(-1, 1)
                ).reshape(1, -1)[0]
                self.data_df[col_name] = self.data_df[col_name].astype(str)
                self.data_df[col_name] = x_np
                self.data_df[col_name] = self.data_df[col_name].astype(int)
        return self.data_df

    def _get_col_value(self, col_idx, value_idx):
        if self.col_types_arr[col_idx] == "C":
            return self.ordinal_encoders_arr[col_idx].inverse_transform([[value_idx]])[0][0]
        return self.bin_names[value_idx]

    def __get_num_combinations(self, dim, num_cols_processed=None):
        avg_unique = mean([len(x) for x in self.unique_vals]) if self.unique_vals else 1
        nc = len(self.data_df.columns)
        if num_cols_processed is not None:
            nc -= num_cols_processed
        return math.comb(max(nc, 0), dim) * pow(avg_unique, 2)

    def __get_is_numeric(self, col_name):
        if is_numeric_dtype(self.data_df[col_name]):
            return True
        non_numeric_count = (
            self.orig_df[col_name]
            .astype(str).str.replace('-', '', regex=False)
            .str.replace('.', '', regex=False)
            .str.isdigit()
            .tolist()
            .count(False)
        )
        return non_numeric_count == 0


# ---------------------------------------------------------------------------
# Higher-dimensional inner-loop helpers (verbatim algorithm from the original)
# ---------------------------------------------------------------------------

def process_inner_loop_3d(obj, i, data_np, num_cols, num_rows, unique_vals,
                          fractions_1d, rare_1d_values, rare_2d_values, divisor):
    num_unique_vals_i = len(unique_vals[i])
    outliers_3d_arr_for_i = [0] * num_rows
    outliers_explanation_arr_for_i = [[]] * num_rows
    rare_arr_for_i = [[[] for _ in range(num_cols)] for _ in range(num_cols)]

    for j in range(i + 1, num_cols - 1):
        num_unique_vals_j = len(unique_vals[j])
        for k in range(j + 1, num_cols):
            num_unique_vals_k = len(unique_vals[k])
            expected_under_uniform = 1.0 / (
                len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k])
            )
            if num_rows * expected_under_uniform < 10:
                continue

            local_rare_arr = [[[False for _ in range(num_unique_vals_k)]
                               for _ in range(num_unique_vals_j)]
                              for _ in range(num_unique_vals_i)]
            for ii in range(num_unique_vals_i):
                if rare_1d_values[i][ii]:
                    continue
                cond1 = (data_np[:, i] == unique_vals[i][ii])
                for jj in range(num_unique_vals_j):
                    if rare_1d_values[j][jj] or rare_2d_values[i][j][ii][jj]:
                        continue
                    cond2 = (data_np[:, j] == unique_vals[j][jj])
                    for kk in range(num_unique_vals_k):
                        if rare_1d_values[k][kk]:
                            continue
                        if rare_2d_values[i][k][ii][kk] or rare_2d_values[j][k][jj][kk]:
                            continue
                        cond3 = (data_np[:, k] == unique_vals[k][kk])
                        rows_all = np.where(cond1 & cond2 & cond3)[0]
                        current_fraction = len(rows_all) / num_rows
                        if obj.check_marginal_probs:
                            expected_given_marginal = (
                                fractions_1d[i][ii] * fractions_1d[j][jj] *
                                fractions_1d[k][kk] * divisor
                            )
                        else:
                            expected_given_marginal = np.inf
                        rare_value_flag = (
                            current_fraction < expected_under_uniform * divisor
                            and current_fraction < expected_given_marginal
                            and current_fraction < 0.01
                        )
                        if rare_value_flag:
                            for r in rows_all:
                                outliers_3d_arr_for_i[r] += 1
                                expl = [[obj.data_df.columns[i], obj.data_df.columns[j], obj.data_df.columns[k]],
                                        [obj._get_col_value(i, ii), obj._get_col_value(j, jj), obj._get_col_value(k, kk)]]
                                if not outliers_explanation_arr_for_i[r]:
                                    outliers_explanation_arr_for_i[r] = [expl]
                                else:
                                    outliers_explanation_arr_for_i[r].append(expl)
                        local_rare_arr[ii][jj][kk] = rare_value_flag
            rare_arr_for_i[j][k] = local_rare_arr
    return rare_arr_for_i, outliers_3d_arr_for_i, outliers_explanation_arr_for_i


def process_inner_loop_4d(obj, i, data_np, num_cols, num_rows, unique_vals,
                          fractions_1d, rare_1d_values, rare_2d_values, rare_3d_values, divisor):
    num_unique_vals_i = len(unique_vals[i])
    outliers_arr = [0] * num_rows
    expl_arr = [[]] * num_rows
    rare_arr_for_i = [[[[] for _ in range(num_cols)] for _ in range(num_cols)] for _ in range(num_cols)]

    for j in range(i + 1, num_cols - 2):
        for k in range(j + 1, num_cols - 1):
            for m in range(k + 1, num_cols):
                expected_under_uniform = 1.0 / (
                    len(unique_vals[i]) * len(unique_vals[j])
                    * len(unique_vals[k]) * len(unique_vals[m])
                )
                if num_rows * expected_under_uniform < 10:
                    continue

                nuv_j, nuv_k, nuv_m = len(unique_vals[j]), len(unique_vals[k]), len(unique_vals[m])
                local_rare_arr = [[[[False for _ in range(nuv_m)] for _ in range(nuv_k)]
                                   for _ in range(nuv_j)] for _ in range(num_unique_vals_i)]
                for ii in range(num_unique_vals_i):
                    if rare_1d_values[i][ii]: continue
                    cond1 = (data_np[:, i] == unique_vals[i][ii])
                    for jj in range(nuv_j):
                        if rare_1d_values[j][jj] or rare_2d_values[i][j][ii][jj]: continue
                        cond2 = (data_np[:, j] == unique_vals[j][jj])
                        for kk in range(nuv_k):
                            if rare_1d_values[k][kk]: continue
                            if rare_2d_values[i][k][ii][kk] or rare_2d_values[j][k][jj][kk]: continue
                            if rare_3d_values[i][j][k][ii][jj][kk]: continue
                            cond3 = (data_np[:, k] == unique_vals[k][kk])
                            for mm in range(nuv_m):
                                if rare_1d_values[m][mm]: continue
                                if rare_2d_values[i][m][ii][mm] or rare_2d_values[j][m][jj][mm] or rare_2d_values[k][m][kk][mm]: continue
                                if rare_3d_values[i][j][m][ii][jj][mm] or rare_3d_values[i][k][m][ii][kk][mm] or rare_3d_values[j][k][m][jj][kk][mm]: continue
                                cond4 = (data_np[:, m] == unique_vals[m][mm])
                                rows_all = np.where(cond1 & cond2 & cond3 & cond4)[0]
                                current_fraction = len(rows_all) / num_rows
                                if obj.check_marginal_probs:
                                    expected_given_marginal = (
                                        fractions_1d[i][ii] * fractions_1d[j][jj]
                                        * fractions_1d[k][kk] * fractions_1d[m][mm] * divisor
                                    )
                                else:
                                    expected_given_marginal = np.inf
                                rare_value_flag = (
                                    current_fraction < expected_under_uniform * divisor
                                    and current_fraction < expected_given_marginal
                                    and current_fraction < 0.01
                                )
                                if rare_value_flag:
                                    for r in rows_all:
                                        outliers_arr[r] += 1
                                        expl = [[obj.data_df.columns[i], obj.data_df.columns[j],
                                                 obj.data_df.columns[k], obj.data_df.columns[m]],
                                                [obj._get_col_value(i, ii), obj._get_col_value(j, jj),
                                                 obj._get_col_value(k, kk), obj._get_col_value(m, mm)]]
                                        if not expl_arr[r]:
                                            expl_arr[r] = [expl]
                                        else:
                                            expl_arr[r].append(expl)
                                local_rare_arr[ii][jj][kk][mm] = rare_value_flag
                rare_arr_for_i[j][k][m] = local_rare_arr
    return rare_arr_for_i, outliers_arr, expl_arr


def process_inner_loop_5d(obj, i, data_np, num_cols, num_rows, unique_vals,
                          fractions_1d, rare_1d_values, rare_2d_values,
                          rare_3d_values, rare_4d_values, divisor):
    num_unique_vals_i = len(unique_vals[i])
    outliers_arr = [0] * num_rows
    expl_arr = [[]] * num_rows
    rare_arr_for_i = [[[[[] for _ in range(num_cols)] for _ in range(num_cols)]
                       for _ in range(num_cols)] for _ in range(num_cols)]

    for j in range(i + 1, num_cols - 3):
        for k in range(j + 1, num_cols - 2):
            for m in range(k + 1, num_cols - 1):
                for n in range(m + 1, num_cols):
                    eu = 1.0 / (
                        len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k])
                        * len(unique_vals[m]) * len(unique_vals[n])
                    )
                    if num_rows * eu < 10:
                        continue
                    nuv_j, nuv_k, nuv_m, nuv_n = (len(unique_vals[j]), len(unique_vals[k]),
                                                  len(unique_vals[m]), len(unique_vals[n]))
                    local_rare_arr = [[[[[False for _ in range(nuv_n)] for _ in range(nuv_m)]
                                        for _ in range(nuv_k)] for _ in range(nuv_j)]
                                      for _ in range(num_unique_vals_i)]
                    for ii in range(num_unique_vals_i):
                        if rare_1d_values[i][ii]: continue
                        cond1 = (data_np[:, i] == unique_vals[i][ii])
                        for jj in range(nuv_j):
                            if rare_1d_values[j][jj] or rare_2d_values[i][j][ii][jj]: continue
                            cond2 = (data_np[:, j] == unique_vals[j][jj])
                            for kk in range(nuv_k):
                                if rare_1d_values[k][kk]: continue
                                if rare_2d_values[i][k][ii][kk] or rare_2d_values[j][k][jj][kk]: continue
                                if rare_3d_values[i][j][k][ii][jj][kk]: continue
                                cond3 = (data_np[:, k] == unique_vals[k][kk])
                                for mm in range(nuv_m):
                                    if rare_1d_values[m][mm]: continue
                                    if rare_2d_values[i][m][ii][mm] or rare_2d_values[j][m][jj][mm] or rare_2d_values[k][m][kk][mm]: continue
                                    if rare_3d_values[i][j][m][ii][jj][mm] or rare_3d_values[i][k][m][ii][kk][mm] or rare_3d_values[j][k][m][jj][kk][mm]: continue
                                    cond4 = (data_np[:, m] == unique_vals[m][mm])
                                    for nn in range(nuv_n):
                                        if rare_1d_values[n][nn]: continue
                                        if (rare_2d_values[i][n][ii][nn] or rare_2d_values[j][n][jj][nn]
                                                or rare_2d_values[k][n][kk][nn] or rare_2d_values[m][n][mm][nn]):
                                            continue
                                        if (rare_3d_values[i][j][n][ii][jj][nn] or rare_3d_values[i][k][n][ii][kk][nn]
                                                or rare_3d_values[i][m][n][ii][mm][nn] or rare_3d_values[j][k][n][jj][kk][nn]
                                                or rare_3d_values[j][m][n][jj][mm][nn] or rare_3d_values[k][m][n][kk][mm][nn]):
                                            continue
                                        if (rare_4d_values[i][j][k][m][ii][jj][kk][mm]
                                                or rare_4d_values[i][j][k][n][ii][jj][kk][nn]
                                                or rare_4d_values[i][j][m][n][ii][jj][mm][nn]
                                                or rare_4d_values[i][k][m][n][ii][kk][mm][nn]
                                                or rare_4d_values[j][k][m][n][jj][kk][mm][nn]):
                                            continue
                                        cond5 = (data_np[:, n] == unique_vals[n][nn])
                                        rows_all = np.where(cond1 & cond2 & cond3 & cond4 & cond5)[0]
                                        current_fraction = len(rows_all) / num_rows
                                        if obj.check_marginal_probs:
                                            expected_given_marginal = (
                                                fractions_1d[i][ii] * fractions_1d[j][jj] * fractions_1d[k][kk]
                                                * fractions_1d[m][mm] * fractions_1d[n][nn] * divisor
                                            )
                                        else:
                                            expected_given_marginal = np.inf
                                        rare_value_flag = (
                                            current_fraction < eu * divisor
                                            and current_fraction < expected_given_marginal
                                            and current_fraction < 0.01
                                        )
                                        if rare_value_flag:
                                            for r in rows_all:
                                                outliers_arr[r] += 1
                                                expl = [[obj.data_df.columns[i], obj.data_df.columns[j],
                                                         obj.data_df.columns[k], obj.data_df.columns[m],
                                                         obj.data_df.columns[n]],
                                                        [obj._get_col_value(i, ii), obj._get_col_value(j, jj),
                                                         obj._get_col_value(k, kk), obj._get_col_value(m, mm),
                                                         obj._get_col_value(n, nn)]]
                                                if not expl_arr[r]:
                                                    expl_arr[r] = [expl]
                                                else:
                                                    expl_arr[r].append(expl)
                                        local_rare_arr[ii][jj][kk][mm][nn] = rare_value_flag
                    rare_arr_for_i[j][k][m][n] = local_rare_arr
    return rare_arr_for_i, outliers_arr, expl_arr


def process_inner_loop_6d(obj, i, data_np, num_cols, num_rows, unique_vals,
                          fractions_1d, rare_1d_values, rare_2d_values,
                          rare_3d_values, rare_4d_values, rare_5d_values, divisor):
    num_unique_vals_i = len(unique_vals[i])
    outliers_arr = [0] * num_rows
    expl_arr = [[]] * num_rows
    rare_arr_for_i = [[[[[[] for _ in range(num_cols)] for _ in range(num_cols)]
                        for _ in range(num_cols)] for _ in range(num_cols)]
                      for _ in range(num_cols)]

    for j in range(i + 1, num_cols - 4):
        for k in range(j + 1, num_cols - 3):
            for m in range(k + 1, num_cols - 2):
                for n in range(m + 1, num_cols - 1):
                    for p in range(n + 1, num_cols):
                        eu = 1.0 / (
                            len(unique_vals[i]) * len(unique_vals[j]) * len(unique_vals[k])
                            * len(unique_vals[m]) * len(unique_vals[n]) * len(unique_vals[p])
                        )
                        if num_rows * eu < 10:
                            continue
                        nuv = (len(unique_vals[j]), len(unique_vals[k]),
                               len(unique_vals[m]), len(unique_vals[n]), len(unique_vals[p]))
                        local_rare_arr = [[[[[[False for _ in range(nuv[4])] for _ in range(nuv[3])]
                                             for _ in range(nuv[2])] for _ in range(nuv[1])]
                                           for _ in range(nuv[0])] for _ in range(num_unique_vals_i)]
                        for ii in range(num_unique_vals_i):
                            if rare_1d_values[i][ii]: continue
                            cond1 = (data_np[:, i] == unique_vals[i][ii])
                            for jj in range(nuv[0]):
                                if rare_1d_values[j][jj] or rare_2d_values[i][j][ii][jj]: continue
                                cond2 = (data_np[:, j] == unique_vals[j][jj])
                                for kk in range(nuv[1]):
                                    if rare_1d_values[k][kk]: continue
                                    if rare_2d_values[i][k][ii][kk] or rare_2d_values[j][k][jj][kk]: continue
                                    if rare_3d_values[i][j][k][ii][jj][kk]: continue
                                    cond3 = (data_np[:, k] == unique_vals[k][kk])
                                    for mm in range(nuv[2]):
                                        if rare_1d_values[m][mm]: continue
                                        if (rare_2d_values[i][m][ii][mm] or rare_2d_values[j][m][jj][mm]
                                                or rare_2d_values[k][m][kk][mm]):
                                            continue
                                        if (rare_3d_values[i][j][m][ii][jj][mm]
                                                or rare_3d_values[i][k][m][ii][kk][mm]
                                                or rare_3d_values[j][k][m][jj][kk][mm]):
                                            continue
                                        cond4 = (data_np[:, m] == unique_vals[m][mm])
                                        for nn in range(nuv[3]):
                                            if rare_1d_values[n][nn]: continue
                                            if (rare_2d_values[i][n][ii][nn] or rare_2d_values[j][n][jj][nn]
                                                    or rare_2d_values[k][n][kk][nn]
                                                    or rare_2d_values[m][n][mm][nn]):
                                                continue
                                            if (rare_3d_values[i][j][n][ii][jj][nn]
                                                    or rare_3d_values[i][k][n][ii][kk][nn]
                                                    or rare_3d_values[i][m][n][ii][mm][nn]
                                                    or rare_3d_values[j][k][n][jj][kk][nn]
                                                    or rare_3d_values[j][m][n][jj][mm][nn]
                                                    or rare_3d_values[k][m][n][kk][mm][nn]):
                                                continue
                                            if rare_4d_values[i][j][k][m][ii][jj][kk][mm]: continue
                                            cond5 = (data_np[:, n] == unique_vals[n][nn])
                                            for pp in range(nuv[4]):
                                                if rare_1d_values[p][pp]: continue
                                                if (rare_2d_values[i][p][ii][pp] or rare_2d_values[j][p][jj][pp]
                                                        or rare_2d_values[k][p][kk][pp]
                                                        or rare_2d_values[m][p][mm][pp]
                                                        or rare_2d_values[n][p][nn][pp]):
                                                    continue
                                                cond6 = (data_np[:, p] == unique_vals[p][pp])
                                                rows_all = np.where(cond1 & cond2 & cond3 & cond4 & cond5 & cond6)[0]
                                                current_fraction = len(rows_all) / num_rows
                                                if obj.check_marginal_probs:
                                                    expected_given_marginal = (
                                                        fractions_1d[i][ii] * fractions_1d[j][jj]
                                                        * fractions_1d[k][kk] * fractions_1d[m][mm]
                                                        * fractions_1d[n][nn] * fractions_1d[p][pp] * divisor
                                                    )
                                                else:
                                                    expected_given_marginal = np.inf
                                                rare_value_flag = (
                                                    current_fraction < eu * divisor
                                                    and current_fraction < expected_given_marginal
                                                    and current_fraction < 0.01
                                                )
                                                if rare_value_flag:
                                                    for r in rows_all:
                                                        outliers_arr[r] += 1
                                                        expl = [
                                                            [obj.data_df.columns[i], obj.data_df.columns[j],
                                                             obj.data_df.columns[k], obj.data_df.columns[m],
                                                             obj.data_df.columns[n], obj.data_df.columns[p]],
                                                            [obj._get_col_value(i, ii), obj._get_col_value(j, jj),
                                                             obj._get_col_value(k, kk), obj._get_col_value(m, mm),
                                                             obj._get_col_value(n, nn), obj._get_col_value(p, pp)],
                                                        ]
                                                        if not expl_arr[r]:
                                                            expl_arr[r] = [expl]
                                                        else:
                                                            expl_arr[r].append(expl)
                                                local_rare_arr[ii][jj][kk][mm][nn][pp] = rare_value_flag
                        rare_arr_for_i[j][k][m][n][p] = local_rare_arr
    return rare_arr_for_i, outliers_arr, expl_arr


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def flatten(arr):
    while True:
        if len(arr) == 0:
            return arr
        if not any(1 for x in arr if isinstance(x, list)):
            return arr
        arr = tuple(i for row in arr for i in row)
