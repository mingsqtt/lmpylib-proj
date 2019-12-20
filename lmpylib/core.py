import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import itertools


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


def barplot(x, y=None, width=0.5, color=None, xlab=None, ylab=None, title=None, sort_by="", ascending=False,
            horizontal=False, show=True):
    if (sort_by != "") and (sort_by != "count") and (sort_by != "label") and (sort_by != "x") and (sort_by != "y"):
        raise Exception("sort_by must be either 'count' or 'label' or 'x' or 'y'")
    if sort_by == "x":
        sort_by = "label"
    if sort_by == "y":
        sort_by = "count"

    if horizontal:
        ascending = not ascending

    plt.style.use("ggplot")
    if y is None:
        if (type(x) == list) or (type(x) == np.ndarray) or (type(x) == pd.Series):
            grouped = _summarize_categorical(x, include_na_only_if_exist=True, sort_by=sort_by, ascending=ascending)
            if horizontal:
                plt.barh(grouped.index, grouped.Count, width, color=color)
            else:
                plt.bar(grouped.index, grouped.Count, width, color=color)
        elif type(x) == pd.DataFrame:
            if len(x.dtypes) > 1:
                if sort_by != "":
                    df = pd.DataFrame({"x": x.iloc[:, 0], "y": x.iloc[:, 1]}).sort_values(
                        by="y" if sort_by == "count" else "x", ascending=ascending)
                    if horizontal:
                        plt.barh(df.x, df.y, width, color=color)
                    else:
                        plt.bar(df.x, df.y, width, color=color)
                elif horizontal:
                    plt.barh(x.iloc[:, 0], x.iloc[:, 1], width, color=color)
                else:
                    plt.bar(x.iloc[:, 0], x.iloc[:, 1], width, color=color)
            elif (str(x.dtypes[0]).find("float") >= 0) | (str(x.dtypes[0]).find("int") >= 0):
                labels = [""] * len(x.index)
                for i, item in enumerate(x.index):
                    if type(item) == tuple:
                        labels[i] = "/".join(item)
                    else:
                        labels[i] = item

                if sort_by != "":
                    df = pd.DataFrame({"x": labels, "y": x.iloc[:, 0]}).sort_values(
                        by="y" if sort_by == "count" else "x", ascending=ascending)
                    if horizontal:
                        plt.barh(df.x, df.y, width, color=color)
                    else:
                        plt.bar(df.x, df.y, width, color=color)
                elif horizontal:
                    plt.barh(labels, x.iloc[:, 0], width, color=color)
                else:
                    plt.bar(labels, x.iloc[:, 0], width, color=color)
            else:
                raise Exception(
                    "Supported DataFrame formats: {col0: label, col1: value}, or grouped DataFrame {index: label, col0: value}")
        else:
            raise Exception("Unsupported type {} for x".format(type(x)))
    else:
        if sort_by != "":
            df = pd.DataFrame({"x": x, "y": y}).sort_values(by="y" if sort_by == "count" else "x", ascending=ascending)
            if horizontal:
                plt.barh(df.x, df.y, width, color=color)
            else:
                plt.bar(df.x, df.y, width, color=color)
        elif horizontal:
            plt.barh(x, y, width, color=color)
        else:
            plt.bar(x, y, width, color=color)
    if xlab is not None:
        plt.xlabel(xlab)
    if ylab is not None:
        plt.ylabel(ylab)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def cartesian_dataframe(df1, df2):
    rows = itertools.product(df1.iterrows(), df2.iterrows())

    df = pd.DataFrame(left.append(right) for (_, left), (_, right) in rows)
    return df.reset_index(drop=True)


