import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools
import math


def _summarize_categorical(data, include_na_only_if_exist=False, sort_by="count", ascending=True):
    na_count = sum([1 for val in data if pd.isna(val)])
    group = pd.DataFrame({"Value": data, "Count": [1] * len(data)}).groupby(by="Value", sort=False).count()
    if sort_by == "count":
        group = group.sort_values(by="Count", ascending=ascending)
    elif (sort_by == "label") or (sort_by == "index") or (sort_by == "text"):
        group = group.sort_index(ascending=ascending)
    group.index = group.index.astype("str")
    if (not include_na_only_if_exist) or (na_count > 0):
        group = group.append(pd.DataFrame({"Count": na_count}, index=["NA"]), ignore_index=False)
    return group


def _summarize_numeric(data, as_rows=False):
    without_na = np.array([val for val in data if (val is not None) and (not pd.isna(val))])

    if len(without_na) == 0:
        if as_rows:
            return pd.DataFrame({
                "Stats": [None, None, None, None, None, None, len(data) - len(without_na)]},
                index=["Min", "Q1", "Median", "Mean", "Q3", "Max", "NA"])
        else:
            return pd.DataFrame({
                "Min": [None],
                "Q1": [None],
                "Median": [None],
                "Mean": [None],
                "Q3": [None],
                "Max": [None],
                "NA": [len(data) - len(without_na)]
            }, index=[""])

    if as_rows:
        return pd.DataFrame({
            "Stats": [np.min(without_na), np.quantile(without_na, 0.25), np.median(without_na), np.mean(without_na),
                      np.quantile(without_na, 0.75), np.max(without_na), len(data) - len(without_na)]},
            index=["Min", "Q1", "Median", "Mean", "Q3", "Max", "NA"])
    else:
        return pd.DataFrame({
            "Min": [np.min(without_na)],
            "Q1": [np.quantile(without_na, 0.25)],
            "Median": [np.median(without_na)],
            "Mean": [np.mean(without_na)],
            "Q3": [np.quantile(without_na, 0.75)],
            "Max": [np.max(without_na)],
            "NA": [len(data) - len(without_na)]
        }, index=[""])


def _summarize_datetime(data, as_rows=False):
    without_na = np.array([val for val in data if (val is not None) and (not pd.isna(val))])

    if len(without_na) == 0:
        if as_rows:
            return pd.DataFrame({
                "Stats": [None, None, None, None, None, None, len(data) - len(without_na)]},
                index=["Min", "Q1", "Median", "Mean", "Q3", "Max", "NA"])
        else:
            return pd.DataFrame({
                "Min": [None],
                "Q1": [None],
                "Median": [None],
                "Mean": [None],
                "Q3": [None],
                "Max": [None],
                "NA": [len(data) - len(without_na)]
            }, index=[""])

    without_na = without_na.astype("datetime64[ns]").astype(np.int64)
    if as_rows:
        return pd.DataFrame({
            "Stats": [np.min(without_na).astype('datetime64[ns]'),
                      np.quantile(without_na, 0.25).astype('datetime64[ns]'),
                      np.median(without_na).astype('datetime64[ns]'), np.mean(without_na).astype('datetime64[ns]'),
                      np.quantile(without_na, 0.75).astype('datetime64[ns]'),
                      np.max(without_na).astype('datetime64[ns]'), len(data) - len(without_na)]},
            index=["Min", "Q1", "Median", "Mean", "Q3", "Max", "NA"])
    else:
        return pd.DataFrame({
            "Min": [np.min(without_na).astype('datetime64[ns]')],
            "Q1": [np.quantile(without_na, 0.25).astype('datetime64[ns]')],
            "Median": [np.median(without_na).astype('datetime64[ns]')],
            "Mean": [np.mean(without_na).astype('datetime64[ns]')],
            "Q3": [np.quantile(without_na, 0.75).astype('datetime64[ns]')],
            "Max": [np.max(without_na).astype('datetime64[ns]')],
            "NA": [len(data) - len(without_na)]
        }, index=[""])


def _is_categorical(val):
    return isinstance(val, bool) | ((not pd.isna(val)) & (not isinstance(val, (int, float))))


def _is_datetime(val):
    return (str(type(val)) == "<class 'datetime.datetime'>") | (str(type(val)) == "<class 'datetime.date'>")


def _is_numeric(val, consider_datetime=False):
    if consider_datetime:
        return _is_datetime(val) or (not _is_categorical(val))
    else:
        return (not _is_datetime(val)) and (not _is_categorical(val))


def _is_numeric_array(arr, consider_datetime=False):
    if type(arr) == np.ndarray:
        type_name = str(arr.dtype)
        if consider_datetime:
            return (type_name.find("int") >= 0) or (type_name.find("float") >= 0) or (type_name.find("datetime") >= 0)
        else:
            return (type_name.find("int") >= 0) or (type_name.find("float") >= 0)
    elif type(arr) == list:
        if len(arr) > 0:
            return _is_numeric(arr[0], consider_datetime) and _is_numeric(arr[-1], consider_datetime)
        else:
            return False
    else:
        raise Exception("Only ndarray is allowed")


def summary(data, is_numeric=None, print_only=None, auto_combine_result=True, _numeric_as_rows=False):
    is_datetime = False
    type_data = type(data)
    if (type_data == list) or (type_data == np.ndarray) or (type_data == pd.Series):
        if is_numeric is None:
            if type_data == list:
                if np.any([_is_datetime(val) for val in data]):
                    is_datetime = True
                else:
                    is_categorical = np.any([_is_categorical(val) for val in data])
            elif str(data.dtype).find("date") >= 0:
                is_datetime = True
            elif str(data.dtype) == "category":
                is_numeric = False
            elif (str(data.dtype).find("int") >= 0) or (str(data.dtype).find("float") >= 0):
                is_numeric = True
            else:
                is_numeric = not np.any([_is_categorical(val) for val in data])

        if is_datetime:
            sum_return = _summarize_datetime(data, _numeric_as_rows)
        elif is_numeric:
            sum_return = _summarize_numeric(data, _numeric_as_rows)
        else:
            sum_return = _summarize_categorical(data)

        if (print_only is not None) & (print_only == True):
            print(sum_return)
        else:
            return sum_return
    elif type_data == pd.DataFrame:
        sum_return = {}
        numeric_summaries = None
        datetime_summaries = None
        for colname in data.columns:
            summ = summary(data[colname])
            if auto_combine_result & (type(summ) == pd.DataFrame) & (
            np.all(summ.columns.to_list() == ["Min", "Q1", "Median", "Mean", "Q3", "Max", "NA"])):
                summ.index = [colname]
                if str(summ.dtypes[0]).find("date") >= 0:
                    if datetime_summaries is None:
                        datetime_summaries = summ
                    else:
                        datetime_summaries = datetime_summaries.append(summ, ignore_index=False)
                else:
                    if numeric_summaries is None:
                        numeric_summaries = summ
                    else:
                        numeric_summaries = numeric_summaries.append(summ, ignore_index=False)
            else:
                sum_return[colname] = summ

        if numeric_summaries is not None:
            if numeric_summaries.shape[0] > 1:
                sum_return["NumericColumns"] = numeric_summaries
            else:
                sum_return[numeric_summaries.index[0]] = numeric_summaries
                numeric_summaries.index = [""]
        if datetime_summaries is not None:
            if datetime_summaries.shape[0] > 1:
                sum_return["DatetimeColumns"] = datetime_summaries
            else:
                sum_return[datetime_summaries.index[0]] = datetime_summaries
                datetime_summaries.index = [""]

        if ((print_only is None) & (len(sum_return) > 1)) | (print_only == True):
            for key in sum_return:
                print("<< {} >>".format(key))
                print(sum_return[key])
                print()
        elif len(sum_return) == 1:
            return list(sum_return.values())[0]
        else:
            return sum_return
    else:
        print("Unsupported type {}".format(type_data))


def structure(data_frame, group_by_type=False, sort_by_type=False, print_only=True):
    if group_by_type | sort_by_type:
        if print_only & group_by_type:
            summ = pd.DataFrame({"column": data_frame.columns.to_list(), "type": [str(t) for t in data_frame.dtypes]})
            for tp in summ.groupby(by="type").indices:
                print("{}:".format(tp))
                print(summ.column.loc[summ.type == tp].to_list())
                print()
        elif print_only & sort_by_type:
            summ = pd.DataFrame({"column": data_frame.columns.to_list(), "type": [str(t) for t in data_frame.dtypes], "index": range(len(data_frame.dtypes))}).sort_values(["type", "index"])
            names = summ.column.to_list()
            max_len = max([len(n) for n in names])
            for i, row in summ.iterrows():
                print("{}  {}  {}".format(str(row["index"]).ljust(int(np.log10(len(names))) + 1), row["column"].ljust(max_len), row["type"]))
        else:
            return pd.DataFrame(
                {"column": data_frame.columns.to_list(), "type": [str(t) for t in data_frame.dtypes], "index": range(len(data_frame.dtypes))}).sort_values(
                by=["type", "index"])
    else:
        if print_only:
            names = data_frame.columns.to_list()
            max_len = max([len(n) for n in names])
            for i, tup in enumerate(zip(names, [str(t) for t in data_frame.dtypes])):
                print("{}  {}  {}".format(str(i).ljust(int(np.log10(len(names))) + 1), tup[0].ljust(max_len), tup[1]))
        else:
            return pd.DataFrame({"column": data_frame.columns.to_list(), "type": [str(t) for t in data_frame.dtypes]})


def hist(x, bins=None, density=False, range=None, color=None, xlab=None, ylab=None, title=None, show=True):
    plt.style.use("ggplot")
    plt.hist(x, bins=bins, density=density, color=color, range=range)
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def auto_figure_size(ncol, nrow, adjust_top=1.0, adjust_bottom=0.0, adjust_hspace=0.2, adjust_wspace=0.2, title_height=None, horizontal=False, print=False):
    fig_width = None
    if ncol == 1:
        fig_width = 6
    elif ncol == 2:
        fig_width = 13
    elif ncol > 2:
        fig_width = 17

    ax_width = np.round((fig_width - 0.5) / (ncol + adjust_wspace * (ncol)), 3)
    if horizontal:
        ax_height = np.round(ax_width * (5 / 7.5), 1)
    else:
        ax_height = np.round(ax_width * (5 / 8.5), 1)
    calc_top = None
    if title_height is None:
        fig_height = np.round(
            (nrow * ax_height + adjust_hspace * ax_height * (nrow - 1)) / (adjust_top - adjust_bottom), 1)
    else:
        fig_height = np.round(
            (nrow * ax_height + adjust_hspace * ax_height * (nrow - 1) + title_height) / (1 - adjust_bottom), 1)
        calc_top = (fig_height - title_height) / fig_height

    if print:
        print("ax size: {} at ratio {}".format((ax_width, ax_height), ax_width / ax_height))
        print("figure_inches={}".format((fig_width, fig_height)))
        print("subplots_adjust_top={}".format(calc_top))
    return (fig_width, fig_height), calc_top


def _try_map(keys_or_indices, mapper, default_map_to_str=True):
    if (type(mapper) == list) or (type(mapper) == np.ndarray) or (type(mapper) == dict):
        if type(keys_or_indices) == pd.Series:
            return ["None" if k is None else mapper[k] for k in keys_or_indices.values]
        else:
            return ["None" if k is None else mapper[k] for k in keys_or_indices]
    elif default_map_to_str:
        return [str(k) for k in keys_or_indices]
    else:
        return keys_or_indices


def _bar_ticks(n, bar_width, n_subbar=1, i_subbar=None):
    if n_subbar == 1:
        if bar_width <= 1:
            return np.linspace(0, (n - 1), n, dtype=int)
        else:
            adjust_width = bar_width + np.round(np.log(bar_width), 1)
            return np.linspace(0, (n-1)*adjust_width, n)
    else:
        if bar_width*n_subbar <= 1:
            adjust_width = bar_width*n_subbar
        else:
            adjust_width = bar_width*n_subbar + np.round(np.log(bar_width*n_subbar), 1)

        if i_subbar is not None:
            return np.linspace(bar_width*i_subbar, bar_width*i_subbar + (n - 1) * adjust_width, n)
        else:
            return np.linspace(0, (n-1)*adjust_width, n) + np.round(bar_width*n_subbar/2 - bar_width/2, 1)


def _barplot_grouped(x, x_col, y_col, group_col, group_position, width, color, x_scale_label_mapper,
                     groups_sort_by_value, groups_ascending, sort_by, ascending, group_label_mapper, horizontal,
                         legend_loc, legend_title):
    x_labels = []
    x_ticks = []

    x_vals = x.iloc[:, x_col].unique()
    if groups_sort_by_value is not None:
        if groups_sort_by_value:
            temp = x.iloc[:, [y_col, group_col]].groupby(
                by=x.columns.values[group_col]).sum().sort_values(by=x.columns.values[y_col],
                                                                  ascending=groups_ascending)
            group_vals = temp.index.values
        else:
            group_vals = x.iloc[:, group_col].sort_values(ascending=groups_ascending).unique()
    else:
        group_vals = x.iloc[:, group_col].unique()
    group_lbls = _try_map(group_vals, group_label_mapper)

    aggregated = x.iloc[:, [x_col, y_col, group_col]].groupby(
        by=[x.columns[group_col], x.columns[x_col]]).sum()

    df = pd.DataFrame({"x": x_vals})
    for group_val in group_vals:
        df = df.merge(aggregated.loc[group_val, :], how="left", left_on="x", right_index=True)
    df.columns = ["x"] + ["y_{}".format(g) for g in range(len(group_vals))]
    df.iloc[:, 1:] = df.iloc[:, 1:].fillna(0)
    df["y"] = df.apply(lambda r: np.sum(r[1:]), axis=1).values

    if sort_by != "":
        df = df.sort_values(by="y" if sort_by == "count" else "x", ascending=ascending)
        x_vals = df.x.values

    x_labels = _try_map(x_vals, x_scale_label_mapper)
    max_y = None

    if group_position == "stack":
        x_ticks = _bar_ticks(len(x_vals), width)
        bottom = np.zeros(len(x_vals))
        for grp, group_val in enumerate(group_vals):
            y_vals = df.iloc[:, grp + 1].values
            if horizontal:
                plt.barh(x_ticks, y_vals, width, color=color, left=bottom, label=group_lbls[grp])
            else:
                plt.bar(x_ticks, y_vals, width, color=color, bottom=bottom, label=group_lbls[grp])
            bottom = bottom + y_vals
        max_y = np.nanmax(df.y.values)

    elif group_position == "dodge":
        for grp, group_val in enumerate(group_vals):
            x_ticks = _bar_ticks(len(x_vals), width, n_subbar=len(group_vals), i_subbar=grp)
            y_vals = df.iloc[:, grp + 1].values
            if horizontal:
                plt.barh(x_ticks, y_vals, width, color=color, label=group_lbls[grp])
            else:
                plt.bar(x_ticks, y_vals, width, color=color, label=group_lbls[grp])
        x_ticks = _bar_ticks(len(x_vals), width, n_subbar=len(group_vals))
        max_y = np.nanmax(df.iloc[:, 1:-1])

    if horizontal:
        plt.xlim([0, max_y * 1.05])
    else:
        plt.ylim([0, max_y * 1.05])

    if legend_loc is not None:
        plt.legend(title=legend_title, facecolor="white", loc=legend_loc, prop={'size': 10}, fontsize=8, ncol=1)

    return x_ticks, x_labels


def barplot(x, y=None, width=0.5, color=None, xlab=None, ylab=None, title=None, sort_by="", ascending=False,
            horizontal=False,  x_scale_rotation=0, x_scale_label_mapper=None,
            group_by=None, group_position="stack", groups_sort_by_value=True, groups_ascending=False, group_label_mapper=None,
            legend_title=None, legend_loc="upper right", show=True):

    if (sort_by != "") and (sort_by != "count") and (sort_by != "label") and (sort_by != "x") and (sort_by != "y"):
        raise Exception("sort_by must be either 'count' or 'label' or 'x' or 'y'")
    if sort_by == "x":
        sort_by = "label"
    if sort_by == "y":
        sort_by = "count"

    if horizontal:
        ascending = not ascending

    plt.style.use("ggplot")

    x_labels = []
    x_ticks = []

    if y is None:
        # if x is 1-D categorical data
        if (type(x) == list) or (type(x) == np.ndarray) or (type(x) == pd.Series):
            grouped = _summarize_categorical(x, include_na_only_if_exist=True, sort_by=sort_by, ascending=ascending)
            x_labels = _try_map(grouped.index, x_scale_label_mapper)
            x_ticks = _bar_ticks(len(x_labels), width)
            if horizontal:
                plt.barh(x_ticks, grouped.Count, width, color=color)
            else:
                plt.bar(x_ticks, grouped.Count, width, color=color)

        # if x is a DataFrame
        elif type(x) == pd.DataFrame:
            # if x is a normal unaggregated DataFrame with 2 or more columns
            # (col_1 will be the category, col_2 will be aggregated as count)
            # if group_by is specified, it has to be specified as column index or column name
            if len(x.dtypes) > 1:
                group_col = None
                x_col = 0
                y_col = 1
                x_vals = x.iloc[:, x_col].values
                y_vals = x.iloc[:, y_col].values

                if (group_by is not None) and (group_by != ""):
                    if type(group_by) == str:
                        group_col = np.argwhere(x.columns.values == group_by)
                        if (len(group_col) != 1) or (group_col[0][0] == -1):
                            raise Exception("group_by column '{}' doesn't exist".format(group_by))
                        else:
                            group_col = group_col[0][0]
                    elif type(group_by) == int:
                        group_col = group_by
                    else:
                        raise Exception("group_by must be a column name or column index")
                    if group_col == 0:
                        x_col = 1
                        y_col = 2
                    elif group_col == 1:
                        y_col = 2

                if group_col is None:
                    aggregated = x.iloc[:, [x_col, y_col]].groupby(by=x.columns[x_col]).sum()
                    x_vals = aggregated.index.values
                    y_vals = aggregated.iloc[:, x_col].values

                    x_ticks = _bar_ticks(len(x_vals), width)

                    if sort_by != "":
                        df = pd.DataFrame({"x": x_vals, "y": y_vals}).sort_values(
                            by="y" if sort_by == "count" else "x", ascending=ascending)
                        x_labels = _try_map(df.x, x_scale_label_mapper)
                        if horizontal:
                            plt.barh(x_ticks, df.y, width, color=color)
                        else:
                            plt.bar(x_ticks, df.y, width, color=color)
                    elif horizontal:
                        x_labels = _try_map(x_vals, x_scale_label_mapper)
                        plt.barh(x_ticks, y_vals, width, color=color)
                    else:
                        x_labels = _try_map(x_vals, x_scale_label_mapper)
                        plt.bar(x_ticks, y_vals, width, color=color)

                else:
                    x_ticks, x_labels = _barplot_grouped(x, x_col, y_col, group_col, group_position, width, color,
                                                         x_scale_label_mapper,
                                                         groups_sort_by_value, groups_ascending, sort_by, ascending,
                                                         group_label_mapper,
                                                         horizontal, legend_loc, legend_title)

            # if x is a group_by DataFrame with only 1 aggregated numeric column
            elif (str(x.dtypes[0]).find("float") >= 0) or (str(x.dtypes[0]).find("int") >= 0):
                # dual-indices
                if (type(x.index[0]) == tuple) and (len(x.index[0]) == 2):
                    if (group_by is not None) and (group_by != ""):
                        group_col = 0

                        if type(group_by) == str:
                            group_col = np.argwhere(np.array(x.index.names) == group_by)
                            if (len(group_col) != 1) or (group_col[0][0] == -1):
                                raise Exception("group_by index '{}' doesn't exist".format(group_by))
                            else:
                                group_col = group_col[0][0]
                        elif type(group_by) == int:
                            group_col = group_by
                            if group_col >= len(x.index.names):
                                raise Exception("Invalid group_by index '{}'".format(group_by))
                        else:
                            raise Exception("group_by must be a index name or index number")

                        df = pd.DataFrame({
                            "group": [tup[group_col] for tup in x.index],
                            "x": [tup[1 if group_col == 0 else 0] for tup in x.index],
                            "y": x.iloc[:, 0]})

                        x_ticks, x_labels = _barplot_grouped(df, 1, 2, 0, group_position, width, color,
                                                             x_scale_label_mapper,
                                                             groups_sort_by_value, groups_ascending, sort_by, ascending,
                                                             group_label_mapper,
                                                             horizontal, legend_loc, legend_title)
                    else:
                        barplot2(x, width=width, color=color, xlab=xlab, ylab=ylab, title=title,
                                                      sort_by=sort_by, ascending=ascending, horizontal=horizontal,
                                                      y_scale_range="fixed",
                                                      x_scale_rotation=x_scale_rotation,
                                                      x_scale_label_mapper=x_scale_label_mapper, show=show)
                        return

                # tri-indices with group_by
                elif (type(x.index[0]) == tuple) and (len(x.index[0]) == 3) and (group_by is not None) and (
                                group_by != ""):
                    barplot2(x, width=width, color=color, xlab=xlab, ylab=ylab, title=title,
                             sort_by=sort_by, ascending=ascending, horizontal=horizontal,
                             y_scale_range="fixed",
                             x_scale_rotation=x_scale_rotation,
                             x_scale_label_mapper=x_scale_label_mapper,
                             group_position=group_position, groups_sort_by_value=groups_sort_by_value,
                             groups_ascending=groups_ascending,
                             group_label_mapper=group_label_mapper,
                             legend_title=legend_title, legend_loc=legend_loc, show=show)
                    return

                # single-index or multiple-indices
                else:
                    labels = [""] * len(x.index)
                    x_ticks = _bar_ticks(len(labels), width)
                    for i, indices in enumerate(x.index):
                        # multiple-indices
                        if type(indices) == tuple:
                            labels[i] = "/".join([str(v) for v in indices])
                        # single-index
                        else:
                            labels[i] = indices

                    if sort_by != "":
                        df = pd.DataFrame({"x": labels, "y": x.iloc[:, 0]}).sort_values(
                            by="y" if sort_by == "count" else "x", ascending=ascending)
                        x_labels = _try_map(df.x, x_scale_label_mapper)
                        if horizontal:
                            plt.barh(x_ticks, df.y, width, color=color)
                        else:
                            plt.bar(x_ticks, df.y, width, color=color)
                    elif horizontal:
                        x_labels = _try_map(labels, x_scale_label_mapper)
                        plt.barh(x_ticks, x.iloc[:, 0], width, color=color)
                    else:
                        x_labels = _try_map(labels, x_scale_label_mapper)
                        plt.bar(x_ticks, x.iloc[:, 0], width, color=color)

            else:
                raise Exception(
                    "Supported DataFrame formats: {col0: label, col1: value}, or grouped DataFrame {index: label, col0: value}")
        else:
            raise Exception("Unsupported type {} for x".format(type(x)))

    # if x is 1-D categorical data, and y is 1-D numeric
    elif ((type(x) == list) or (type(x) == np.ndarray) or (type(x) == pd.Series)) and \
            ((type(y) == list) or (type(y) == np.ndarray) or (type(y) == pd.Series)):
        df = pd.DataFrame({"x": x, "y": y})
        x_labels = _try_map(df.x, x_scale_label_mapper)
        x_ticks = _bar_ticks(len(x_labels), width)

        if sort_by != "":
            df = df.sort_values(by="y" if sort_by == "count" else "x", ascending=ascending)
            x_labels = _try_map(df.x, x_scale_label_mapper)
            if horizontal:
                plt.barh(x_ticks, df.y, width, color=color)
            else:
                plt.bar(x_ticks, df.y, width, color=color)
        elif horizontal:
            plt.barh(x_ticks, df.y, width, color=color)
        else:
            plt.bar(x_ticks, df.y, width, color=color)

    else:
        raise Exception("Unsupported type {} for x, or {} for y".format(type(x), type(y)))

    if horizontal:
        plt.yticks(x_ticks, x_labels, rotation=x_scale_rotation)
    else:
        plt.xticks(x_ticks, x_labels, rotation=x_scale_rotation)

    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def barplot2(x, width=0.5, color=None, xlab=None, ylab=None, title=None, sort_by="", ascending=False,
                 horizontal=False, figure_inches=None, y_scale_range=None, x_scale_rotation=0,
                 x_scale_label_mapper=None, facet_label_mapper=None, subplots_ncol=4, subplots_adjust=None,
                 subplots_adjust_top=1.0, subplots_adjust_bottom=0.0, subplots_adjust_hspace=0.2,
                 subplots_adjust_wspace=0.2,
                 group_position="stack", groups_sort_by_value=True, groups_ascending=False,
                 group_label_mapper=None,
                 legend_title=None, legend_loc="upper right", show=False):

    if (sort_by != "") and (sort_by != "count") and (sort_by != "label") and (sort_by != "x") and (sort_by != "y"):
        raise Exception("sort_by must be either 'count' or 'label' or 'x' or 'y'")
    if sort_by == "x":
        sort_by = "label"
    if sort_by == "y":
        sort_by = "count"

    if horizontal:
        ascending = not ascending

    plt.style.use("ggplot")

    if (type(x) == pd.DataFrame) and (len(x.dtypes) == 1) and (type(x.index[0]) == tuple) and \
            ((len(x.index[0]) == 2) or (len(x.index[0]) == 3)) and \
            ((str(x.dtypes[0]).find("float") >= 0) or (str(x.dtypes[0]).find("int") >= 0)):

        facet_index = 0
        x_index = 1 if len(x.index[0]) == 2 else 2
        group_index = 1 if len(x.index[0]) == 3 else None

        df_indecis = x.index.values
        all_facet_vals = [tup[facet_index] for tup in df_indecis]
        facet_vals = pd.Series(all_facet_vals).unique()
        all_x_vals = [tup[x_index] for tup in df_indecis]
        x_vals = pd.Series(all_x_vals).unique()
        group_vals = None
        global_max_y = None
        if group_index is not None:
            group_vals = pd.Series([tup[group_index] for tup in df_indecis]).unique()
            global_max_y = np.nanmax(pd.DataFrame({
                "f": all_facet_vals,
                "x": all_x_vals,
                "y": x.iloc[:, 0]}).groupby(["f", "x"]).sum()["y"])

        if subplots_ncol > len(facet_vals):
            subplots_ncol = len(facet_vals)
        subplot_nrow = int(math.ceil(len(facet_vals) / subplots_ncol))

        if group_vals is None:
            df_facet_spread = pd.DataFrame({"x": x_vals})
            for i, facet in enumerate(facet_vals):
                df_facet_spread = df_facet_spread.merge(x.loc[facet, :], how="left", left_on="x", right_index=True)
            df_facet_spread.columns = ["x"] + ["y_{}".format(i) for i in range(len(facet_vals))]
            df_facet_spread.iloc[:, 1:] = df_facet_spread.iloc[:, 1:].fillna(0).values
        else:
            df_facet_spread = cartesian_dataframe(pd.DataFrame({"group": group_vals}), pd.DataFrame({"x": x_vals}))
            for i, facet in enumerate(facet_vals):
                df_facet_spread = df_facet_spread.merge(x.loc[facet, :], how="left", left_on=["group", "x"], right_index=True)
            df_facet_spread.columns = ["group", "x"] + ["f_{}".format(i) for i in range(len(facet_vals))]
            df_facet_spread.iloc[:, 2:] = df_facet_spread.iloc[:, 2:].fillna(0).values

        fig, axes = plt.subplots(subplot_nrow, subplots_ncol)
        n_hidden_axes = subplot_nrow*subplots_ncol % len(facet_vals)
        if (n_hidden_axes > 0) and (subplot_nrow > 1):
            for hid in range(subplots_ncol-n_hidden_axes, subplots_ncol):
                axes[-1, hid].axis("off")

        if subplots_adjust is not None:
            plt.subplots_adjust(top=subplots_adjust[0], bottom=subplots_adjust[1], hspace=subplots_adjust[2],
                                wspace=subplots_adjust[3])
            subplots_adjust_top = subplots_adjust[0]
            subplots_adjust_bottom = subplots_adjust[1]
            subplots_adjust_hspace = subplots_adjust[2]
            subplots_adjust_wspace = subplots_adjust[3]
        else:
            if subplots_adjust_top is not None:
                plt.subplots_adjust(top=subplots_adjust_top)
            if subplots_adjust_bottom is not None:
                plt.subplots_adjust(bottom=subplots_adjust_bottom)
            if subplots_adjust_hspace is not None:
                plt.subplots_adjust(hspace=subplots_adjust_hspace)
            if subplots_adjust_wspace is not None:
                plt.subplots_adjust(wspace=subplots_adjust_wspace)

        if figure_inches is not None:
            fig.set_size_inches(figure_inches[0], figure_inches[1])
        else:
            if (title is not None) and (title != ""):
                auto_size, new_top = auto_figure_size(subplots_ncol, subplot_nrow,
                                              1,
                                              subplots_adjust_bottom,
                                              subplots_adjust_hspace,
                                              subplots_adjust_wspace,
                                              title_height=0.5, horizontal=horizontal)
                plt.subplots_adjust(top=new_top)
            else:
                auto_size, _ = auto_figure_size(subplots_ncol, subplot_nrow,
                                              subplots_adjust_top,
                                              subplots_adjust_bottom,
                                              subplots_adjust_hspace,
                                              subplots_adjust_wspace, horizontal=horizontal)
            fig.set_size_inches(auto_size[0], auto_size[1])

        plots = list()
        group_lbls = None
        for i, facet in enumerate(facet_vals):
            ax_row = int(i / subplots_ncol)
            ax_col = i % subplots_ncol
            ax = None
            if (subplot_nrow > 1) and (subplots_ncol > 1):
                ax = axes[ax_row, ax_col]
            elif subplot_nrow == 1:
                ax = axes[ax_col]
            elif subplots_ncol == 1:
                ax = axes[ax_row]

            # without group
            if group_index is None:
                df = df_facet_spread.iloc[:, [0, i + 1]]

                if sort_by != "":
                    df = df.sort_values(by="y_{}".format(i) if sort_by == "count" else "x", ascending=ascending)
                x_labels = _try_map(df.x, x_scale_label_mapper)

                if horizontal:
                    ax.barh(x_labels, df.iloc[:, 1], width, color=color)
                else:
                    ax.bar(x_labels, df.iloc[:, 1], width, color=color)

                ax.set_xlabel(_try_map([facet], facet_label_mapper)[0])

                if y_scale_range is not None:
                    if horizontal:
                        if (type(y_scale_range) == tuple) or (type(y_scale_range) == list):
                            ax.set_xlim([y_scale_range[0], y_scale_range[1]])
                        elif (type(y_scale_range) == str) and (y_scale_range == "fixed"):
                            max_y = np.nanmax(x.iloc[:, 0].values)
                            ax.set_xlim([0, max_y * 1.05])
                        else:
                            ax.set_xlim([0, y_scale_range])
                    else:
                        if (type(y_scale_range) == tuple) or (type(y_scale_range) == list):
                            ax.set_ylim([y_scale_range[0], y_scale_range[1]])
                        elif (type(y_scale_range) == str) and (y_scale_range == "fixed"):
                            max_y = np.nanmax(x.iloc[:, 0].values)
                            ax.set_ylim([0, max_y * 1.05])
                        else:
                            ax.set_ylim([0, y_scale_range])

                if horizontal:
                    if (ax_col == 0) or (sort_by == "count"):
                        if (x_scale_rotation is not None) and (x_scale_rotation > 0):
                            ax.set_yticklabels(x_labels, rotation=x_scale_rotation)
                        else:
                            ax.set_yticklabels(x_labels)
                    else:
                        ax.set_yticklabels([])
                else:
                    if (ax_row == subplot_nrow - 1) or (
                            (ax_row == subplot_nrow - 2) and (ax_col >= (subplots_ncol - n_hidden_axes))) or \
                            (sort_by == "count"):
                        if (x_scale_rotation is not None) and (x_scale_rotation > 0):
                            ax.set_xticklabels(x_labels, rotation=x_scale_rotation)
                        else:
                            ax.set_xticklabels(x_labels)
                    else:
                        ax.set_xticklabels([])

            # with group
            else:
                df_current_facet = df_facet_spread.iloc[:, [0, 1, i + 2]]

                if groups_sort_by_value is not None:
                    if groups_sort_by_value:
                        temp = df_current_facet.iloc[:, [2, 0]].groupby(by="group").sum().sort_values(by="f_{}".format(i),
                                                                                                    ascending=groups_ascending)
                        group_vals = temp.index.values
                    else:
                        group_vals = df_current_facet.iloc[:, 0].sort_values(ascending=groups_ascending).unique()
                else:
                    group_vals = df_current_facet.iloc[:, 0].unique()
                group_lbls = _try_map(group_vals, group_label_mapper)

                df = pd.DataFrame({"x": x_vals})
                for group_val in group_vals:
                    df = df.merge(df_current_facet.loc[df_current_facet["group"] == group_val, ["x", "f_{}".format(i)]], how="left", left_on="x", right_on="x")
                df.columns = ["x"] + ["y_{}".format(g) for g in range(len(group_vals))]
                df.iloc[:, 1:] = df.iloc[:, 1:].fillna(0)
                df["y"] = df.apply(lambda r: np.sum(r[1:]), axis=1).values

                if sort_by != "":
                    df = df.sort_values(by="y" if sort_by == "count" else "x", ascending=ascending)
                    x_vals = df.x.values

                x_labels = _try_map(x_vals, x_scale_label_mapper)
                x_ticks = []
                max_y = None

                if group_position == "stack":
                    x_ticks = _bar_ticks(len(x_vals), width)
                    bottom = np.zeros(len(x_vals))
                    for grp, group_val in enumerate(group_vals):
                        y_vals = df.iloc[:, grp + 1].values
                        if horizontal:
                            plots.append(ax.barh(x_ticks, y_vals, width, color=color, left=bottom))
                        else:
                            plots.append(ax.bar(x_ticks, y_vals, width, color=color, bottom=bottom))
                        bottom = bottom + y_vals
                    max_y = np.nanmax(df.y.values)

                elif group_position == "dodge":
                    for grp, group_val in enumerate(group_vals):
                        x_ticks = _bar_ticks(len(x_vals), width, n_subbar=len(group_vals), i_subbar=grp)
                        y_vals = df.iloc[:, grp + 1].values
                        if horizontal:
                            plots.append(ax.barh(x_ticks, y_vals, width, color=color))
                        else:
                            plots.append(ax.bar(x_ticks, y_vals, width, color=color))
                    x_ticks = _bar_ticks(len(x_vals), width, n_subbar=len(group_vals))
                    max_y = np.nanmax(df.iloc[:, 1:-1])

                ax.set_xlabel(_try_map([facet], facet_label_mapper)[0])

                if y_scale_range is not None:
                    if horizontal:
                        if (type(y_scale_range) == tuple) or (type(y_scale_range) == list):
                            ax.set_xlim([y_scale_range[0], y_scale_range[1]])
                        elif (type(y_scale_range) == str) and (y_scale_range == "fixed"):
                            ax.set_xlim([0, global_max_y * 1.05])
                        else:
                            ax.set_xlim([0, y_scale_range])
                    else:
                        if (type(y_scale_range) == tuple) or (type(y_scale_range) == list):
                            ax.set_ylim([y_scale_range[0], y_scale_range[1]])
                        elif (type(y_scale_range) == str) and (y_scale_range == "fixed"):
                            ax.set_ylim([0, global_max_y * 1.05])
                        else:
                            ax.set_ylim([0, y_scale_range])
                else:
                    if horizontal:
                        ax.set_xlim([0, max_y * 1.05])
                    else:
                        ax.set_ylim([0, max_y * 1.05])

                if horizontal:
                    ax.set_yticks(x_ticks)
                    if (ax_col == 0) or (sort_by == "count"):
                        if (x_scale_rotation is not None) and (x_scale_rotation > 0):
                            ax.set_yticklabels(x_labels, rotation=x_scale_rotation)
                        else:
                            ax.set_yticklabels(x_labels)
                    else:
                        ax.set_yticklabels([])
                else:
                    ax.set_xticks(x_ticks)
                    if (ax_row == subplot_nrow - 1) or (
                            (ax_row == subplot_nrow - 2) and (ax_col >= (subplots_ncol - n_hidden_axes))) or \
                            (sort_by == "count"):
                        if (x_scale_rotation is not None) and (x_scale_rotation > 0):
                            ax.set_xticklabels(x_labels, rotation=x_scale_rotation)
                        else:
                            ax.set_xticklabels(x_labels)
                    else:
                        ax.set_xticklabels([])

        if (group_index is not None) and (legend_loc is not None):
            fig.legend(tuple(plots), tuple(group_lbls), title=legend_title, facecolor="white", loc=legend_loc, prop={'size': 10}, fontsize=8,
                       ncol=1)

        if xlab is not None:
            if horizontal:
                fig.text(0.04, 0.5, xlab, va='center', rotation='vertical')
            else:
                fig.text(0.5, 0.04, xlab, ha='center')
        if ylab is not None:
            if horizontal:
                fig.text(0.5, 0.04, ylab, ha='center')
            else:
                fig.text(0.04, 0.5, ylab, va='center', rotation='vertical')
        if title is not None:
            fig.suptitle(title, size=16)
        if show:
            fig.show()
    else:
        raise Exception(
            "x must be a DataFrame with 1 numeric column, 2 indices (without group_by) or 3 indices (with group_by).")


def plot(x, y=None, style="solid", width=1.0, color=None, marker=None, marker_size=None, xlab=None, ylab=None, title=None,
         x_scale_ticks=None, x_scale_rotation=0, show=True):

    plt.style.use("ggplot")

    x_vals = None
    y_vals = None

    if y is None:
        # if x is 1-D numeric data
        if (type(x) == list) or (type(x) == np.ndarray) or (type(x) == pd.Series):
            x_vals = np.array(range(len(x)), dtype=int)
            y_vals = x

        # if x is a DataFrame
        elif type(x) == pd.DataFrame:
            # if x is a normal DataFrame with 2 or more columns (only col_1 and col_2 will be used)
            if len(x.dtypes) >= 2:
                x_vals = x.iloc[:, 0].values
                y_vals = x.iloc[:, 1].values
                if (not _is_numeric_array(x_vals)) or (len(x.iloc[:, 0].unique()) < len(x_vals)):
                    raise Exception("Consider calling plot2() after aggregating the data frame by '{}'".format(x.columns[0]))

            # if x is a single-column DataFrame
            elif len(x.dtypes) == 1:
                y_vals = x.iloc[:, 0].values
                if _is_numeric_array(x.index.values, True):
                    x_vals = x.index.values
                elif (type(x.index[0]) == tuple) and (len(x.index[0]) == 2) and (_is_numeric(x.index[0][1], True)):
                    # call plot2
                    pass
                else:
                    x_vals = np.array(range(len(x)), dtype=int)

            else:
                raise Exception(
                    "Supported DataFrame formats: {col0: numeric, col1: numeric}")
        else:
            raise Exception("Unsupported type {} for x".format(type(x)))

    # if both x and y are 1-D numeric
    elif ((type(x) == list) or (type(x) == np.ndarray) or (type(x) == pd.Series)) and \
        ((type(y) == list) or (type(y) == np.ndarray) or (type(y) == pd.Series)):
        x_vals = x.values if type(x) == pd.Series else np.array(x)
        y_vals = y.values if type(y) == pd.Series else np.array(y)

    else:
        raise Exception("Unsupported type {} for x, or {} for y".format(type(x), type(y)))

    if (style == "point") or (style == "scatter"):
        plt.scatter(x_vals, y_vals, color=color, marker=marker, s=marker_size)
    else:
        plt.plot(x_vals, y_vals, linewidth=width, linestyle=style, color=color, marker=marker, markersize=marker_size)

    if (type(x_scale_ticks) == list) or (type(x_scale_ticks) == np.ndarray):
        if _is_numeric_array(x_scale_ticks, consider_datetime=True):
            plt.xticks(x_scale_ticks, x_scale_ticks)
        else:
            tick_vals = np.linspace(np.min(x_vals), np.max(x_vals), len(x_scale_ticks), dtype=x_vals.dtype)
            plt.xticks(tick_vals, x_scale_ticks)
    elif type(x_scale_ticks) == dict:
        plt.xticks(list(x_scale_ticks.keys()), list(x_scale_ticks.values()))

    if x_scale_rotation > 0:
        plt.xticks(rotation=x_scale_rotation)

    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def scatter(x, y=None, color=None, marker=None, marker_size=5, xlab=None, ylab=None, title=None,
         x_scale_ticks=None, x_scale_rotation=0, show=True):
    plot(x, y=y, style="scatter", color=color, marker=marker, marker_size=marker_size,
         xlab=xlab, ylab=ylab, title=title,
         x_scale_ticks=x_scale_ticks, x_scale_rotation=x_scale_rotation, show=show)


def cartesian_dataframe(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())

    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)






