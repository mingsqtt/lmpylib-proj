import numpy as np
import pandas as pd
import re
import hashlib
from datetime import timedelta, datetime

EMAIL_REGEX = "(?i)(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|\"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*\")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])"
SHORT_MONTH_NAMES = np.array(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
LONG_MONTH_NAMES = np.array(["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"])


def search_pattern(text, pattern, take_parts=0):
    if type(pattern) == str:
        pattern = re.compile(pattern)

    if text is None:
        return None
    elif (type(text) == list) or (type(text) == np.ndarray) or (type(text) == pd.Series):
        results = [pattern.search(s) for s in text]
        if type(take_parts) == int:
            return [r.group(take_parts) if r is not None else None for r in results]
        elif (type(take_parts) == list) or (type(take_parts) == np.ndarray):
            df = pd.DataFrame()
            for i, p in enumerate(take_parts):
                df["p" + str(p)] = [r.group(p) if r is not None else None for r in results]
            return df
        else:
            return results
    else:
        assert type(text) == str, "text must be string or array"

        results = pattern.search(text)
        if results is None:
            return None
        elif type(take_parts) == int:
            return results.group(take_parts)
        elif (type(take_parts) == list) or (type(take_parts) == np.ndarray):
            return [results.group(p) for p in take_parts]
        else:
            return results


def search_time_pattern(text, search_prepositions=["before", "bfr", "after", "aft", "between", "from", "frm", "no later than", "no earlier than", "at", "to", "by", "till", "not", "n't"], including_named_time=["early morning", "morning", "late afternoon", "afternoon", "noon", "late evening", "evening", "midnight", "tonight", "night"]):
    hour_part = "(00|01|02|03|04|05|06|07|08|09|10|11|12|13|14|15|16|17|18|19|20|21|22|23|\d)"
    min_part = "(?::([0-5]?[0-9])(?::[0-5][0-9])?)"
    dot_min_part = "(?:.([0-5]?[0-9])(?:.[0-5][0-9])?)"
    full_min_part = "([0-5][0-9])"
    am_part = "(?:\s*[\./\s]?[\(@]?\s?(a\.m|am|p\.m|pm|morning|afternoon|noon|tonight|night|evening|hours|hour|hrs|hr)(?:\s?\))?)"
    prep_part_opt = "(?:({})\s*)?".format("|".join(search_prepositions))
    prep_part_opt_for_named = "(?:({})\s+)?".format("|".join(search_prepositions))
    prep_part = "(?:({})\s*)".format("|".join(search_prepositions))
    start_part, end_part = "(?:^|[\s\.\?\(\[\"',;:!@<>\*\|\+])", "(?:[\s\.\?\)\]\-\"',;\:!<>/\*\|]|$)"

    cols_5 = list(range(1, 6))
    cols_4 = list(range(1, 5))

    if (type(text) == list) or (type(text) == np.ndarray) or (type(text) == pd.Series):
        df = pd.DataFrame({
            "text": [""] * len(text),
            "prep": [""] * len(text),
            "hour": [""] * len(text),
            "minute": [""] * len(text),
            "am": [""] * len(text)
        })

        hr_min_am_pattern = "(?i){}({}{}{}{}){}".format(start_part, prep_part_opt, hour_part, min_part, am_part, end_part)
        temp = search_pattern(text, hr_min_am_pattern, cols_5)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        hr_min_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part_opt, hour_part, min_part, end_part)
        temp = search_pattern(text, hr_min_pattern, cols_4)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        hr_min_am_pattern = "(?i){}({}{}{}{}){}".format(start_part, prep_part_opt, hour_part, full_min_part, am_part, end_part)
        temp = search_pattern(text, hr_min_am_pattern, cols_5)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        prep_hr_min_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part, hour_part, full_min_part, end_part)
        temp = search_pattern(text, prep_hr_min_pattern, cols_4)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        hr_min_am_pattern = "(?i){}({}{}{}{}){}".format(start_part, prep_part_opt, hour_part, dot_min_part, am_part, end_part)
        temp = search_pattern(text, hr_min_am_pattern, cols_5)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        prep_hr_min_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part, hour_part, dot_min_part, end_part)
        temp = search_pattern(text, prep_hr_min_pattern, cols_4)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        hours_and_miniutes_pattern = "(?i){}({}{}(?: hour|hour| hr|hr)\(?s\)?(?: and)? ([0-5]?[0-9])(?: minute|minute| min|min)\(?s\)?){}".format(
            start_part, prep_part_opt, hour_part, end_part)
        temp = search_pattern(text, hours_and_miniutes_pattern, cols_4)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        hr_am_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part_opt, hour_part, am_part, end_part)
        temp = search_pattern(text, hr_am_pattern, cols_4)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        df.iloc[ind, 0] = temp.iloc[ind, 0].values
        df.iloc[ind, 1] = temp.iloc[ind, 1].values
        df.iloc[ind, 2] = temp.iloc[ind, 2].values
        df.iloc[ind, 4] = temp.iloc[ind, 3].values

        if (including_named_time is not None) and (len(including_named_time) > 0):
            named_pattern = "(?i){}({}(?:in the |in |the )?({})){}".format(start_part, prep_part_opt_for_named, "|".join(including_named_time), end_part)
            temp = search_pattern(text, named_pattern, [1, 2, 3])
            ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
            df.iloc[ind, 0] = temp.iloc[ind, 0].values
            df.iloc[ind, 1] = temp.iloc[ind, 1].values
            df.iloc[ind, 4] = temp.iloc[ind, 2].values

        return df
    else:
        hr_min_am_pattern = "(?i){}({}{}{}{}){}".format(start_part, prep_part_opt, hour_part, min_part, am_part, end_part)
        temp = search_pattern(text, hr_min_am_pattern, cols_5)
        if (temp is not None) and (temp[0] is not None):
            return temp

        hr_min_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part_opt, hour_part, min_part, end_part)
        temp = search_pattern(text, hr_min_pattern, cols_4)
        if (temp is not None) and (temp[0] is not None):
            return temp + [None]

        hr_min_am_pattern = "(?i){}({}{}{}{}){}".format(start_part, prep_part_opt, hour_part, full_min_part, am_part, end_part)
        temp = search_pattern(text, hr_min_am_pattern, cols_5)
        if (temp is not None) and (temp[0] is not None):
            return temp

        prep_hr_min_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part, hour_part, full_min_part, end_part)
        temp = search_pattern(text, prep_hr_min_pattern, cols_4)
        if (temp is not None) and (temp[0] is not None):
            return temp + [None]

        hr_min_am_pattern = "(?i){}({}{}{}{}){}".format(start_part, prep_part_opt, hour_part, dot_min_part, am_part, end_part)
        temp = search_pattern(text, hr_min_am_pattern, cols_5)
        if (temp is not None) and (temp[0] is not None):
            return temp

        prep_hr_min_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part, hour_part, dot_min_part, end_part)
        temp = search_pattern(text, prep_hr_min_pattern, cols_4)
        if (temp is not None) and (temp[0] is not None):
            return temp + [None]

        hours_and_miniutes_pattern = "(?i){}({}{}(?: hour|hour| hr|hr)\(?s\)?(?: and)? ([0-5]?[0-9])(?: minute|minute| min|min)\(?s\)?){}".format(start_part, prep_part_opt,
                                                                                                         hour_part, end_part)
        temp = search_pattern(text, hours_and_miniutes_pattern, cols_4)
        if (temp is not None) and (temp[0] is not None):
            return temp + [None]

        hr_am_pattern = "(?i){}({}{}{}){}".format(start_part, prep_part_opt, hour_part, am_part, end_part)
        temp = search_pattern(text, hr_am_pattern, cols_4)
        if (temp is not None) and (temp[0] is not None):
            return temp[:3] + [None, temp[-1]]

        if (including_named_time is not None) and (len(including_named_time) > 0):
            named_pattern = "(?i){}({}(?:in the |in |the )?({})){}".format(start_part, prep_part_opt_for_named, "|".join(including_named_time), end_part)
            temp = search_pattern(text, named_pattern, [1, 2, 3])
            if (temp is not None) and (temp[0] is not None):
                return temp[:2] + [None, None, temp[-1]]

        return temp


def search_date_pattern(text, search_prepositions=["before", "bfr", "after", "aft", "between", "from", "frm", "no later than", "no earlier than", "as of", "at", "on", "to", "by", "since", "till", "not", "n't"], including_named_date=["today", "tdy", "tonight", "the day after tomorrow", "day after tomorrow", "tomorrow", "tmr", "next day", "last weekend", "this weekend", "next weekend", "weekend", "last (?:monday|(?-i:M)on)", "this (?:monday|(?-i:M)on)", "coming (?:monday|(?-i:M)on)", "next (?:monday|(?-i:M)on)", "(?:monday|(?-i:M)on)", "last (?:tuesday|(?-i:T)ue)", "this (?:tuesday|(?-i:T)ue)", "coming (?:tuesday|(?-i:T)ue)", "next (?:tuesday|(?-i:T)ue)", "(?:tuesday|(?-i:T)ue)", "last (?:wednesday|(?-i:W)ed)", "this (?:wednesday|(?-i:W)ed)", "coming (?:wednesday|(?-i:W)ed)", "next (?:wednesday|(?-i:W)ed)", "(?:wednesday|(?-i:W)ed)", "last (?:thursday|(?-i:T)hu)", "this (?:thursday|(?-i:T)hu)", "coming (?:thursday|(?-i:T)hu)", "next (?:thursday|(?-i:T)hu)", "(?:thursday|(?-i:T)hu)", "last (?:friday|(?-i:F)ri)", "this (?:friday|(?-i:F)ri)", "coming (?:friday|(?-i:F)ri)", "next (?:friday|(?-i:F)ri)", "(?:friday|(?-i:F)ri)", "last (?:saturday|(?-i:S)at)", "this (?:saturday|(?-i:S)at)", "coming (?:saturday|(?-i:S)at)", "next (?:saturday|(?-i:S)at)", "(?:saturday|(?-i:S)at)", "last (?:sunday|(?-i:S)un)", "this (?:sunday|(?-i:S)un)", "coming (?:sunday|(?-i:S)un)", "next (?:sunday|(?-i:S)un)", "(?:sunday|(?-i:S)un)"]):
    day_part = "([0-3]?[0-9])"
    # num_month_part = "(12|11|10|(?:0?[1-9]))"
    cha_month_part = "(" + "|".join(LONG_MONTH_NAMES.tolist() + SHORT_MONTH_NAMES.tolist()) + ")"
    month_part = "(" + "|".join(LONG_MONTH_NAMES.tolist() + SHORT_MONTH_NAMES.tolist()) + "|12|11|10|(?:0?[1-9]))"
    year_part = "((?:19|20)?[0-9]{2})"
    full_year_part = "((?:19|20)[0-9]{2})"
    prep_part_opt = "(?:({})\s*)?".format("|".join(search_prepositions))
    start_part, end_part = "(?:^|[\s\.\?\(\[\"',;:!@<>\*\|\+])", "(?:[\s\.\?\)\]\"',;:!@<>\*\|]|$)"

    dmMy_slash = "(?:{}/{}(?:/{})?)".format(day_part, month_part, year_part)
    dmMy_dot_dash = "(?:{}[\.-]{}[\.-]{})".format(day_part, month_part, year_part)
    dMy_space_dot_slash = "(?:{}(?:st|nd|rd|th)?[\s/\.,]*(?:of\s)?{}[\s/\.,]*{})".format(day_part, cha_month_part, year_part)

    YmMd_slash_dot_dash = "(?:{}[/\.-]{}[/\.-]{})".format(full_year_part, month_part, day_part)
    YMd_space = "(?:(?:{}\s+{}\s+{})(?:st|nd|rd|th)?)".format(full_year_part, cha_month_part, day_part)

    MdY_space_dot_slash = "(?:{}\s+{}(?:st|nd|rd|th)?[\s/\.,]*{})".format(cha_month_part, day_part, full_year_part)

    dM_all = "(?:{}(?:st|nd|rd|th)?[\s/\.\-]*(?:of\s)?{})".format(day_part, cha_month_part)
    Md_all = "(?:{}[\s/\.\-]*{}(?:st|nd|rd|th)?)".format(cha_month_part, day_part)

    MY_space_dot_slash = "(?:{}[\s/\.,]*{})".format(cha_month_part, full_year_part)
    YM_space_dot_slash = "(?:{}[\s/\.,]*{})".format(full_year_part, cha_month_part)

    cols_5_ymd = [1, 2, 3, 4, 5]
    cols_5_dmy = [1, 2, 5, 4, 3]
    cols_5_mdy = [1, 2, 5, 3, 4]
    cols_4_ym = [1, 2, 3, 4]
    cols_4_my = [1, 2, 4, 3]
    cols_4_dm = [1, 2, 4, 3]
    cols_4_md = [1, 2, 3, 4]

    if (type(text) == list) or (type(text) == np.ndarray) or (type(text) == pd.Series):
        df = pd.DataFrame({
            "text": [""] * len(text),
            "prep": [""] * len(text),
            "year": [""] * len(text),
            "month": [""] * len(text),
            "day": [""] * len(text)
        })

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dmMy_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_dmy)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dmMy_dot_dash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_dmy)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, YmMd_slash_dot_dash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_ymd)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, YMd_space, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_ymd)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dMy_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_dmy)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        for c in [0, 1, 2, 3, 4]:
            df.iloc[ind, c] = temp.iloc[ind, c].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, MdY_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_mdy)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        df.iloc[ind, 0] = temp.iloc[ind, 0].values
        df.iloc[ind, 1] = temp.iloc[ind, 1].values
        df.iloc[ind, 2] = temp.iloc[ind, 4].values
        df.iloc[ind, 3] = temp.iloc[ind, 2].values
        df.iloc[ind, 4] = temp.iloc[ind, 3].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dM_all, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_dm)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        df.iloc[ind, 0] = temp.iloc[ind, 0].values
        df.iloc[ind, 1] = temp.iloc[ind, 1].values
        df.iloc[ind, 3] = temp.iloc[ind, 3].values
        df.iloc[ind, 4] = temp.iloc[ind, 2].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, Md_all, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_dm)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        df.iloc[ind, 0] = temp.iloc[ind, 0].values
        df.iloc[ind, 1] = temp.iloc[ind, 1].values
        df.iloc[ind, 3] = temp.iloc[ind, 2].values
        df.iloc[ind, 4] = temp.iloc[ind, 3].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, MY_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_my)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        df.iloc[ind, 0] = temp.iloc[ind, 0].values
        df.iloc[ind, 1] = temp.iloc[ind, 1].values
        df.iloc[ind, 2] = temp.iloc[ind, 2].values
        df.iloc[ind, 3] = temp.iloc[ind, 3].values

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, YM_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_ym)
        ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
        df.iloc[ind, 0] = temp.iloc[ind, 0].values
        df.iloc[ind, 1] = temp.iloc[ind, 1].values
        df.iloc[ind, 2] = temp.iloc[ind, 2].values
        df.iloc[ind, 3] = temp.iloc[ind, 3].values

        if (including_named_date is not None) and (len(including_named_date) > 0):
            named_pattern = "(?i){}({}(?:the )?({})){}".format(start_part, prep_part_opt, "|".join(including_named_date), end_part)
            temp = search_pattern(text, named_pattern, [1, 2, 3])
            ind = np.argwhere(pd.isna(temp.iloc[:, 0].values) == False).flatten()
            df.iloc[ind, 0] = temp.iloc[ind, 0].values
            df.iloc[ind, 1] = temp.iloc[ind, 1].values
            df.iloc[ind, 4] = temp.iloc[ind, 2].values

        return df
    else:
        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dmMy_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_dmy)
        if (temp is not None) and (temp[0] is not None):
            return temp

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dmMy_dot_dash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_dmy)
        if (temp is not None) and (temp[0] is not None):
            return temp

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, YmMd_slash_dot_dash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_ymd)
        if (temp is not None) and (temp[0] is not None):
            return temp

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, YMd_space, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_ymd)
        if (temp is not None) and (temp[0] is not None):
            return temp

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dMy_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_dmy)
        if (temp is not None) and (temp[0] is not None):
            return temp

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, MdY_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_5_mdy)
        if (temp is not None) and (temp[0] is not None):
            return temp

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, dM_all, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_dm)
        if (temp is not None) and (temp[0] is not None):
            return temp[:2] + [None] + temp[2:]

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, Md_all, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_md)
        if (temp is not None) and (temp[0] is not None):
            return temp[:2] + [None] + temp[2:]

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, MY_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_my)
        if (temp is not None) and (temp[0] is not None):
            return temp + [None]

        single_date_pattern = "(?i){}({}{}){}".format(start_part, prep_part_opt, YM_space_dot_slash, end_part)
        temp = search_pattern(text, single_date_pattern, cols_4_ym)
        if (temp is not None) and (temp[0] is not None):
            return temp + [None]

        if (including_named_date is not None) and (len(including_named_date) > 0):
            named_pattern = "(?i){}({}(?:the )?({})){}".format(start_part, prep_part_opt, "|".join(including_named_date), end_part)
            temp = search_pattern(text, named_pattern, [1, 2, 3])
            if (temp is not None) and (temp[0] is not None):
                return temp[:2] + [None, None, temp[-1]]

        return temp


def search_all_dates(text, search_prepositions=["before", "bfr", "after", "aft", "between", "from", "frm", "no later than", "no earlier than", "as of", "at", "on", "to", "by", "since", "till", "-", "–", "~", "not", "n't"], including_named_date=["today", "tdy", "tonight", "the day after tomorrow", "day after tomorrow", "tomorrow", "tmr", "next day", "last weekend", "this weekend", "next weekend", "weekend", "last (?:monday|(?-i:M)on)", "this (?:monday|(?-i:M)on)", "coming (?:monday|(?-i:M)on)", "next (?:monday|(?-i:M)on)", "(?:monday|(?-i:M)on)", "last (?:tuesday|(?-i:T)ue)", "this (?:tuesday|(?-i:T)ue)", "coming (?:tuesday|(?-i:T)ue)", "next (?:tuesday|(?-i:T)ue)", "(?:tuesday|(?-i:T)ue)", "last (?:wednesday|(?-i:W)ed)", "this (?:wednesday|(?-i:W)ed)", "coming (?:wednesday|(?-i:W)ed)", "next (?:wednesday|(?-i:W)ed)", "(?:wednesday|(?-i:W)ed)", "last (?:thursday|(?-i:T)hu)", "this (?:thursday|(?-i:T)hu)", "coming (?:thursday|(?-i:T)hu)", "next (?:thursday|(?-i:T)hu)", "(?:thursday|(?-i:T)hu)", "last (?:friday|(?-i:F)ri)", "this (?:friday|(?-i:F)ri)", "coming (?:friday|(?-i:F)ri)", "next (?:friday|(?-i:F)ri)", "(?:friday|(?-i:F)ri)", "last (?:saturday|(?-i:S)at)", "this (?:saturday|(?-i:S)at)", "coming (?:saturday|(?-i:S)at)", "next (?:saturday|(?-i:S)at)", "(?:saturday|(?-i:S)at)", "last (?:sunday|(?-i:S)un)", "this (?:sunday|(?-i:S)un)", "coming (?:sunday|(?-i:S)un)", "next (?:sunday|(?-i:S)un)", "(?:sunday|(?-i:S)un)"], _index_offset=0):
    if type(text) == str:
        results = list()
        result = search_date_pattern(text, search_prepositions, including_named_date)
        if (result is not None) and (result[0] is not None):
            from_idx = text.find(result[0])
            to_idx = from_idx + len(result[0])

            before_text = text[:from_idx]
            if before_text != "":
                before_results = search_all_dates(before_text, search_prepositions, including_named_date)
                if len(before_results) > 0:
                    results.extend(before_results)

            if len(result) == 4:
                results.append((from_idx + _index_offset, to_idx + _index_offset, result[0], result[1], result[2], result[3]))
            elif len(result) == 5:
                results.append((from_idx + _index_offset, to_idx + _index_offset, result[0], result[1], result[2], result[3], result[4]))

            after_text = text[to_idx:]
            if after_text != "":
                after_results = search_all_dates(after_text, search_prepositions, including_named_date, _index_offset=to_idx+_index_offset)
                if len(after_results) > 0:
                    results.extend(after_results)
        return results
    else:
        raise Exception("text must be string")


def search_all_times(text, search_prepositions=["before", "bfr", "after", "aft", "between", "from", "frm", "no later than", "no earlier than", "at", "to", "by", "till", "-", "–", "~", "not", "n't"], including_named_time=["early morning", "morning", "late afternoon", "afternoon", "noon", "late evening", "evening", "midnight", "tonight", "night"], _index_offset=0):
    if type(text) == str:
        results = list()
        result = search_time_pattern(text, search_prepositions, including_named_time)
        if (result is not None) and (result[0] is not None):
            from_idx = text.find(result[0])
            to_idx = from_idx + len(result[0])

            before_text = text[:from_idx]
            if before_text != "":
                before_results = search_all_times(before_text, search_prepositions, including_named_time)
                if len(before_results) > 0:
                    results.extend(before_results)

            if len(result) == 4:
                results.append((from_idx + _index_offset, to_idx + _index_offset, result[0], result[1], result[2], result[3]))
            elif len(result) == 5:
                results.append((from_idx + _index_offset, to_idx + _index_offset, result[0], result[1], result[2], result[3], result[4]))

            after_text = text[to_idx:]
            if after_text != "":
                after_results = search_all_times(after_text, search_prepositions, including_named_time, _index_offset=to_idx+_index_offset)
                if len(after_results) > 0:
                    results.extend(after_results)
        return results
    else:
        raise Exception("text must be string")


def search_all_datetimes(text):
    datetimes = list()
    all_dates = search_all_dates(text)
    prev_end = 0
    for date_tup in all_dates:
        start = date_tup[0]
        end = date_tup[1]
        if start > prev_end:
            sub_text = text[prev_end:start]
            times = search_all_times(sub_text)
            for time_tup in times:
                new_arr = list(time_tup)
                new_arr[0] += prev_end
                new_arr[1] += prev_end
                datetimes.append(tuple(["time"] + new_arr))
        datetimes.append(tuple(["date"] + list(date_tup)))
        prev_end = end

    if prev_end < len(text):
        sub_text = text[prev_end:]
        times = search_all_times(sub_text)
        for time_tup in times:
            new_arr = list(time_tup)
            new_arr[0] += prev_end
            new_arr[1] += prev_end
            datetimes.append(tuple(["time"] + new_arr))
    return datetimes


def md5(text, encoding="utf-16"):
    m = hashlib.md5()
    m.update(text.encode(encoding))
    return m.digest().hex()


def decode_html(encoded):
    if encoded is None:
        return ""
    else:
        return encoded.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt", ">").replace("&amp;", "&")


def html_to_plain(html, return_segments=False):
    cursor = 0
    html2 = ""
    while cursor < len(html):
        cha = html[cursor]
        if cha == "<":
            remaining = html[cursor+1:]
            closing_ind = remaining.find(">")
            if closing_ind > 0:
                if not (remaining.startswith("span") or remaining.startswith("/span") or remaining.startswith(
                        "font") or remaining.startswith("/font") or remaining.startswith(
                        "strong") or remaining.startswith("/strong")):
                    html2 += html[cursor:cursor + closing_ind + 2]
                cursor += closing_ind + 2
            else:
                html2 += html[cursor:]
                break
        else:
            opening_ind = html[cursor + 1:].find("<")
            if opening_ind > 0:
                html2 += html[cursor:cursor + opening_ind + 1]
                cursor += opening_ind + 1
            elif opening_ind == 0:
                cursor += 1
            else:
                html2 += html[cursor:]
                break

    cursor = 0
    segments = list()
    nnewline = 0
    nparam = 0
    while cursor < len(html2):
        cha = html2[cursor]
        if cha == "<":
            remaining = html2[cursor+1:]
            closing_ind = remaining.find(">")
            if closing_ind > 0:
                if remaining.startswith("div") or remaining.startswith("tr"):
                    nnewline += 1
                elif remaining.startswith("/div") or remaining.startswith("/tr"):
                    if nnewline > 0:
                        segments.append("\n" * nnewline)
                        nnewline = 0
                elif remaining.startswith("br>") or remaining.startswith("br/>") or remaining.startswith("br />"):
                    segments.append("\n")
                elif remaining.startswith("p") or remaining.startswith("h1") or remaining.startswith(
                        "h2") or remaining.startswith("h3") or remaining.startswith("h4") or remaining.startswith("h5"):
                    segments.append("\n")
                    nparam += 1
                elif remaining.startswith("/p") or remaining.startswith("/h1") or remaining.startswith(
                        "/h2") or remaining.startswith("/h3") or remaining.startswith("/h4") or remaining.startswith("/h5"):
                    segments.append("\n"*nparam)
                    nparam = 0
                cursor += closing_ind + 2
            else:
                break
        else:
            opening_ind = html2[cursor + 1:].find("<")
            if opening_ind > 0:
                segments.append(decode_html(html2[cursor:cursor+opening_ind+1]))
                segments.append("\n" * nparam)
                nparam = 0
                cursor += opening_ind + 1
            elif opening_ind == 0:
                cursor += 1
            else:
                segments.append(decode_html(html2[cursor:]))
                break

    if return_segments:
        return segments
    else:
        return "".join(segments).strip('\n')


def get_tagged_class(text, tag_name="p"):
    open_tag = "<{}>".format(tag_name)
    close_tag = "</{}>".format(tag_name)
    a = text.find(open_tag)
    if a > 0:
        b = text.find(close_tag)
        return text[a + len(open_tag):b]
    else:
        return None


def token_casing_feature(token, return_one_hot=True):
    code, desc = 5, "others"
    any_cap_match = re.search("[A-Z]", token)
    if any_cap_match is None:
        code, desc = 1, "lower"
    else:
        any_lower_match = re.search("[a-z]", token)
        if any_lower_match is None:
            code, desc = 2, "cap"
        else:
            title_match = re.search("^[A-Z][^A-Z]+$", token)
            if title_match is not None:
                code, desc = 3, "title"
            else:
                pascal_match = re.search("^[A-Z][^A-Z]*[A-Z]+[^A-Z]+$", token)
                if pascal_match is not None:
                    code, desc = 4, "pascal"

    if return_one_hot:
        vec = np.array([0, 0, 0, 0, 0], dtype=int)
        if code > 0:
            vec[code - 1] = 1
        return vec
    else:
        return code, desc


def token_type_feature(token, return_one_hot=True):
    code, desc = 8, "others"
    only_alpha_match = re.search("^[A-Za-z&]+$", token)
    if only_alpha_match is not None:
        code, desc = 1, "alpha"
    else:
        only_num_match = re.search("^-?\(?(\d{1,3},)*\d+(\.\d+)?\)?$", token)
        if only_num_match is not None:
            code, desc = 2, "number"
        else:
            alpha_num_match = re.search("^\w+$", token)
            if alpha_num_match is not None:
                code, desc = 3, "alpha_number"
            else:
                punc_match = re.search("^[{}<>\[\]\(\),\.\?\:;\"'\|\\\\!@#\$%^&\*_\-=\+]+$", token)
                if punc_match is not None:
                    code, desc = 4, "punc"
                else:
                    connected_match = re.search("^.+(?:-.+)+$", token)
                    if connected_match is not None:
                        code, desc = 5, "connected"
                    else:
                        tuple_match = re.search("^.+(?:/.+)+$", token)
                        if tuple_match is not None:
                            code, desc = 6, "tuple"
                        else:
                            braced_match = re.search("^(\(\w+\))|(\[\w+\])$", token)
                            if braced_match is not None:
                                code, desc = 7, "braced"
    if return_one_hot:
        vec = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=int)
        if code > 0:
            vec[code - 1] = 1
        return vec
    else:
        return code, desc


def remap_tokens_to_text(tokens, original_text, include_end_pos=True, include_end_pos_as_pairs=False, token_mapper={}):
    mapped_positions = list()
    end_positions = list()
    search_from_pos = 0
    for token in tokens:
        mapped_pos = original_text.find(token, search_from_pos)
        if mapped_pos == -1:
            if (token_mapper is not None) and (token_mapper.__contains__(token)):
                token = token_mapper[token]
                mapped_pos = original_text.find(token, search_from_pos)
                if mapped_pos == -1:
                    raise Exception(
                        "tokens and the original text cannot be remapped, because the token '{}' is not found in the original text from position {} onwards:\n{}".format(
                            token, search_from_pos, original_text))
            else:
                raise Exception(
                    "tokens and the original text cannot be remapped, because the token '{}' is not found in the original text from position {} onwards. Consider using token_mapper argument to map the token to the original characters in the original text:\n{}".format(
                        token, search_from_pos, original_text))
        if mapped_pos > search_from_pos:
            space_match = re.search("^\s+$", original_text[search_from_pos:mapped_pos])
            if space_match is None:
                raise Exception(
                    "tokens and the original text cannot be remapped, because there is non-space between position {} and {} of the original text:\n{}".format(
                        search_from_pos, mapped_pos, original_text))
        mapped_positions.append(mapped_pos)
        end_positions.append(mapped_pos + len(token))
        search_from_pos = len(token) + mapped_pos
    if len(original_text) > search_from_pos:
        space_match = re.search("^\s+$", original_text[search_from_pos:])
        if space_match is None:
            raise Exception(
                "tokens and the original text cannot be remapped, because there is non-space from position {} onwards in the original text:\n".format(
                    search_from_pos))

    if include_end_pos:
        if include_end_pos_as_pairs:
            return [(mapped_positions[i], end_positions[i]) for i in range(len(mapped_positions))]
        else:
            return mapped_positions, end_positions
    else:
        return mapped_positions


def span_predictions_to_fragments(predictions, tokens, original_text, span_begin_class=1, span_intermidate_class=2, span_end_class=3,
                                  token_mapper={}, return_pos_pairs=False):
    assert len(predictions) == len(tokens), "predictions and tokens must have same length, but got {} and {}".format(len(predictions),
                                                                                                                     len(tokens))

    remapped_token_start_pos, remapped_token_end_pos = remap_tokens_to_text(tokens, original_text, token_mapper=token_mapper)

    frag_pos = list()
    start_pos, end_pos = None, None
    for t, pred in enumerate(predictions):
        if t == 0:
            if (pred == span_begin_class) or (pred == span_intermidate_class):
                start_pos = remapped_token_start_pos[t]
                end_pos = remapped_token_end_pos[t]
        elif t < len(tokens) - 1:
            if start_pos is not None:
                if (pred == span_intermidate_class) or (pred == span_end_class):
                    end_pos = remapped_token_end_pos[t]
                elif pred == span_begin_class:
                    frag_pos.append((start_pos, end_pos))
                    start_pos = remapped_token_start_pos[t]
                    end_pos = remapped_token_end_pos[t]
                else:
                    frag_pos.append((start_pos, end_pos))
                    start_pos, end_pos = None, None
            else:
                if (pred == span_begin_class) or (pred == span_intermidate_class):
                    start_pos = remapped_token_start_pos[t]
                    end_pos = remapped_token_end_pos[t]
        else:
            if start_pos is not None:
                if (pred == span_intermidate_class) or (pred == span_end_class):
                    end_pos = remapped_token_end_pos[t]
                    frag_pos.append((start_pos, end_pos))
                elif pred == span_begin_class:
                    frag_pos.append((start_pos, end_pos))
                    start_pos = remapped_token_start_pos[t]
                    end_pos = remapped_token_end_pos[t]
                    frag_pos.append((start_pos, end_pos))
                else:
                    frag_pos.append((start_pos, end_pos))
                    start_pos, end_pos = None, None
            else:
                if (pred == span_begin_class) or (pred == span_intermidate_class):
                    start_pos = remapped_token_start_pos[t]
                    end_pos = remapped_token_end_pos[t]
                    frag_pos.append((start_pos, end_pos))

    if not return_pos_pairs:
        return [original_text[pos[0]:pos[1]] for pos in frag_pos]
    else:
        return frag_pos


def get_sequence_labels(text, valid_classes=None, return_spacy_format=False):
    if (text.find("</") > 0) and (text.find(">") > 0):
        if valid_classes is not None:
            valid_classes = set(valid_classes)

        ori_text = ""
        n_char = len(text)
        i, ori_i = 0, 0
        tag_stack = list()
        labels = list()
        while i < n_char:
            ch = text[i]
            if ch == "<":
                ind = text.find(">", i + 1)
                if ind > 0:
                    if (i + 1 < n_char) and (text[i + 1] == "/"):
                        closing_tag = text[(i + 2):ind].upper()
                        if closing_tag.find("<") == -1:
                            if (len(tag_stack) > 0) and (tag_stack[len(tag_stack) - 1][0] == closing_tag):
                                labels.append(tag_stack.pop())
                                i += 3 + len(closing_tag)
                                continue
                            else:
                                raise Exception(
                                    "Expecting opening tag <{}>, but not found in text:\n{}\n\nStack:\n{}".format(closing_tag, text,
                                                                                                                  str(tag_stack)))
                    else:
                        opening_tag = text[(i + 1):ind].upper()
                        if (opening_tag.find("MAILTO") == -1) and (opening_tag.find("HTTP") == -1) and (opening_tag.find("<") == -1):
                            if (valid_classes is not None) and (not valid_classes.__contains__(opening_tag)):
                                raise Exception("<{}> tag is not allowed:\n{}\n\nStack:\n{}".format(opening_tag, text, (
                                    str(tag_stack) if len(tag_stack) > 0 else "")))
                            else:
                                tag_stack.append([opening_tag, ori_i, ori_i, ""])
                                i += 2 + len(opening_tag)
                                continue

            ori_text += ch
            for tag in tag_stack:
                tag[2] += 1
                tag[3] += ch
            i += 1
            ori_i += 1

        if return_spacy_format:
            annotations = list()
            for label in labels:
                annotations.append((label[1], label[2], label[0]))
            return ori_text, {"entities": annotations}
        else:
            return ori_text, labels
    else:
        if return_spacy_format:
            return text, {"entities": []}
        else:
            return text, []


def map_sequence_labels_to_tokens(sequence_labelled_text, target_class, tokenizer_func, return_token_label_pairs=False):
    prompt = "sequence_labelled_text must be tuple of (text_str, label_list). each label must be list of [class_str, from_pos, to_pos, fragment_str]."
    assert len(sequence_labelled_text) == 2, prompt
    assert type(sequence_labelled_text[0]) == str, prompt
    assert type(sequence_labelled_text[1]) == list, prompt
    text = sequence_labelled_text[0]
    tokens = list()
    token_level_labels = list()
    from_pos = 0
    if type(target_class) == list:
        target_class = np.array(target_class)
    for label in sorted(sequence_labelled_text[1], key=lambda entry: entry[1]):
        if type(target_class) == np.ndarray:
            class_index = np.argwhere(target_class == label[0]).flatten()
            if len(class_index) == 0:
                continue
            else:
                class_index = class_index[0]
        elif target_class == label[0]:
            class_index = 0
        else:
            continue

        offset = class_index * 3

        fragment_from_pos, fragment_to_pos = label[1], label[2]

        if fragment_from_pos > from_pos:
            null_tokens = tokenizer_func(text[from_pos:fragment_from_pos])
            if len(null_tokens) > 0:
                tokens.extend(null_tokens)
                token_level_labels.extend([0] * len(null_tokens))

        fragment_tokens = tokenizer_func(text[fragment_from_pos:fragment_to_pos])
        tokens.extend(fragment_tokens)
        if len(fragment_tokens) == 1:
            token_level_labels.append(1 + offset)
        elif len(fragment_tokens) == 2:
            token_level_labels.extend([1 + offset, 3 + offset])
        else:
            token_level_labels.extend([1 + offset] + [2 + offset] * (len(fragment_tokens) - 2) + [3 + offset])

        from_pos = fragment_to_pos

    if from_pos < len(text):
        null_tokens = tokenizer_func(text[from_pos:len(text)])
        if len(null_tokens) > 0:
            tokens.extend(null_tokens)
            token_level_labels.extend([0] * len(null_tokens))

    if return_token_label_pairs:
        pairs = list()
        for i in range(len(tokens)):
            pairs.append((tokens[i], token_level_labels[i]))
        return pairs
    else:
        return tokens, token_level_labels


def my_word_tokenize(text, tokenize_func):
    tokens = list()
    for token in tokenize_func(text):
        if (len(token) > 1) and ((token.find(",") == 0) or (token.find(".") == 0)):
            tokens.append(token[:1])
            tokens.append(token[1:])
        else:
            match = re.search("(.*[A-Za-z]),(\d.*)", token)
            if match is None:
                tokens.append(token)
            else:
                tokens.append(match[1])
                tokens.append(",")
                tokens.append(match[2])
    return tokens


def create_augmented_spacy_ner_data(input_ner_labelled_docs, ner_classes, replacement_data, n_iter, keep_trainable_classes=None):
    ner_classes = np.array(ner_classes)
    if keep_trainable_classes is not None:
        keep_trainable_classes = np.array(keep_trainable_classes)

    docs = list()
    for doc in input_ner_labelled_docs:
        for tup in doc[1]["entities"]:
            if np.any(tup[2] == ner_classes):
                docs.append(doc)
                break

    augmented_data = list()
    for it in range(n_iter):
        for doc in docs:
            processable_labels = list()
            for tup in doc[1]["entities"]:
                processable_labels.append([tup[0], tup[1], tup[2], np.any(tup[2] == ner_classes)])

            new_text = doc[0]
            for t in range(len(processable_labels)):
                lbl = processable_labels[t]
                if not lbl[3]:
                    continue
                cur_start = lbl[0]
                cur_end = lbl[1]
                cur_len = cur_end - cur_start
                cur_class = lbl[2]
                replacement = str(np.random.choice(replacement_data[cur_class], 1)[0])
                new_len = len(replacement)
                new_text = new_text[:cur_start] + replacement + new_text[cur_end:]
                if new_len != cur_len:
                    len_diff = new_len - cur_len
                    for t0 in range(len(processable_labels)):
                        lbl0 = processable_labels[t0]
                        if lbl0[0] >= cur_end:
                            lbl0[0] += len_diff
                        if lbl0[1] >= cur_end:
                            lbl0[1] += len_diff

            new_annotations = list()
            for lbl in processable_labels:
                if (keep_trainable_classes is None) or (np.any(keep_trainable_classes == lbl[2])):
                    new_annotations.append((lbl[0], lbl[1], lbl[2]))
            augmented_data.append((new_text, {"entities": new_annotations}))
    return augmented_data


def eval_seqential_labelling_by_span(doc_actual, doc_pred, partial_match_score=0.5):
    true_pos, false_pos = 0, 0
    n_actual = len(doc_actual)
    n_pred = len(doc_pred)
    if (n_pred > 0) and (n_actual > 0):
        for y_hat in doc_pred:
            y_hat_start = y_hat[0]
            y_hat_end = y_hat[1]
            y_hat_class = y_hat[2]

            for y in doc_actual:
                y_start = y[0]
                y_end = y[1]
                y_class = y[2]
                if y_class == y_hat_class:
                    if (y_start == y_hat_start) and (y_end == y_hat_end):
                        true_pos += 1
                        break
                    elif (y_start <= y_hat_start) and (y_end >= y_hat_end):
                        true_pos += partial_match_score
                        break
            else:
                false_pos += 1

        precision = round(true_pos / n_pred, 3)
        recall = round(true_pos / n_actual, 3)
        if (precision + recall) > 0:
            f1 = round(2 * precision * recall / (precision + recall), 5)
        else:
            f1 = 0
    elif (n_pred == 0) and (n_actual == 0):
        precision = 1
        recall = 1
        f1 = 1
    else:
        precision = 0
        recall = 0
        f1 = 0
    return precision, recall, f1


def eval_seqential_labelling_by_text(doc_actual, doc_pred, partial_match_score=0.5):
    true_pos, false_pos = 0, 0
    n_actual = len(doc_actual)
    n_pred = len(doc_pred)
    if (n_pred > 0) and (n_actual > 0):
        for y_hat in doc_pred:
            y_hat_text = y_hat[0]
            y_hat_class = y_hat[1]

            for y in doc_actual:
                y_text = y[0]
                y_class = y[1]
                if y_class == y_hat_class:
                    if y_text == y_hat_text:
                        true_pos += 1
                        break
                    elif y_text.find(y_hat_text) >= 0:
                        true_pos += partial_match_score
                        break
            else:
                false_pos += 1

        precision = round(true_pos / n_pred, 3)
        recall = round(true_pos / n_actual, 3)
        if (precision + recall) > 0:
            f1 = round(2 * precision * recall / (precision + recall), 5)
        else:
            f1 = 0
    elif (n_pred == 0) and (n_actual == 0):
        precision = 1
        recall = 1
        f1 = 1
    else:
        precision = 0
        recall = 0
        f1 = 0
    return precision, recall, f1


def eval_seqential_labelling(actuals, predicts, partial_match_score=0.5, return_mean=True):
    n_actuals = len(actuals)
    n_predicts = len(predicts)
    assert n_actuals == n_predicts, "Length of actuals and predicts should be same, but got ({}, {})".format(n_actuals, n_predicts)

    if n_actuals > 0:
        by_text = True
        for doc in actuals:
            if len(doc) > 0:
                if (len(doc[0]) == 3) and (type(doc[0][0]) == int):
                    by_text = False
                    break
                elif (len(doc[0]) == 2) and (type(doc[0][0]) == str):
                    by_text = True
                    break
                else:
                    assert 1 == 0, "Invalid labelling, expected (text, class) or (start, end, class)"

        precisions, recalls, f1s = np.zeros(n_actuals), np.zeros(n_actuals), np.zeros(n_actuals)
        for i in range(n_actuals):
            if by_text:
                precision, recall, f1 = eval_seqential_labelling_by_text(actuals[i], predicts[i], partial_match_score)
            else:
                precision, recall, f1 = eval_seqential_labelling_by_span(actuals[i], predicts[i], partial_match_score)
            precisions[i] = precision
            recalls[i] = recall
            f1s[i] = f1
        if return_mean:
            return precisions.mean(), recalls.mean(), f1s.mean()
        else:
            return precisions, recalls, f1s
    else:
        return None, None, None


def tuple_to_datetime(date_tuple=None, time_tuple=None):
    now = datetime.now()
    prep_str = None

    hour = 0
    minute = 0

    if time_tuple is not None:
        if time_tuple[0] == "time":
            assert len(time_tuple) == 8, "time_tuple should have length of 8"
            prep_str = time_tuple[4]
            hour_str = time_tuple[5]
            min_str = time_tuple[6]
            pm_str = time_tuple[7]
        else:
            assert len(time_tuple) == 5, "time_tuple should have length of 5"
            prep_str = time_tuple[1]
            hour_str = time_tuple[2]
            min_str = time_tuple[3]
            pm_str = time_tuple[4]

        if hour_str is not None:
            hour = int(hour_str)

        if min_str is not None:
            minute = int(min_str)

        if pm_str is not None:
            pm_str = pm_str.lower()
            if 0 < hour < 12:
                if (pm_str == "p.m") or (pm_str == "p.m.") or (pm_str == "pm") or (pm_str == "late afternoon") or (pm_str == "afternoon") or (pm_str == "noon") or (
                        pm_str == "tonight") or (pm_str == "night") or (pm_str == "late evening") or (pm_str == "evening"):
                    hour += 12
            elif (hour_str is None) and (min_str is None):
                if (pm_str == "afternoon") or (pm_str == "noon"):
                    hour = 12
                elif pm_str == "late afternoon":
                    hour = 16
                elif (pm_str == "tonight") or (pm_str == "night") or (pm_str == "late evening") or (pm_str == "evening"):
                    hour = 20

    if date_tuple is not None:
        if date_tuple[0] == "date":
            assert len(date_tuple) == 8, "date_tuple should have length of 8"
            if date_tuple[4] is not None:
                prep_str = date_tuple[4]
            year_str = date_tuple[5]
            month_str = date_tuple[6]
            day_str = date_tuple[7]
        else:
            assert len(date_tuple) == 5, "date_tuple should have length of 5"
            if date_tuple[1] is not None:
                prep_str = date_tuple[1]
            year_str = date_tuple[2]
            month_str = date_tuple[3]
            day_str = date_tuple[4]

        if month_str is not None:
            if len(month_str) == 3:
                month = np.argwhere(SHORT_MONTH_NAMES == month_str.lower())[0, 0] + 1
            elif len(month_str) > 3:
                month = np.argwhere(LONG_MONTH_NAMES == month_str.lower())[0, 0] + 1
            else:
                month = int(month_str)
        else:
            month = str(now.month)

        if year_str is None:
            year_str = str(now.year)

        if day_str is None:
            day_str = str(now.day)
        elif len(day_str) > 2:
            day_str = day_str.lower()
            if (day_str == "today") or (day_str == "tdy") or (day_str == "tonight"):
                dt = now
                return datetime(dt.year, dt.month, dt.day, hour, minute, 0), prep_str
            elif (day_str == "the day after tomorrow") or (day_str == "day after tomorrow"):
                dt = now + timedelta(days=2)
                return datetime(dt.year, dt.month, dt.day, hour, minute, 0), prep_str
            elif (day_str == "tomorrow") or (day_str == "tmr"):
                dt = now + timedelta(days=1)
                return datetime(dt.year, dt.month, dt.day, hour, minute, 0), prep_str
            elif day_str == "yesterday":
                dt = now + timedelta(days=-1)
                return datetime(dt.year, dt.month, dt.day, hour, minute, 0), prep_str

        if len(year_str) == 2:
            return datetime(2000 + int(year_str), month, int(day_str), hour, minute, 0), prep_str
        else:
            return datetime(int(year_str), month, int(day_str), hour, minute, 0), prep_str

    elif (hour != 0) or (minute != 0):
        return datetime(1970, 1, 1, hour, minute, 0), prep_str

    else:
        return None, None

