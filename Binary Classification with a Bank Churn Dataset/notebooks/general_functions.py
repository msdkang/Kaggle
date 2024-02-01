import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats import shapiro, normaltest, kstest, anderson
from statsmodels.graphics.gofplots import qqplot
import folium
import branca.colormap as cm


# DATA VALIDATION
class data_validation:
    def get_data_schema(df: pd.DataFrame) -> dict:
        '''
        `get_data_schema` generates data schema of a given dataframe.

        Parameters
        ----------
        df: `DataFrame`
            input dataset    

        Returns
        -------
        schema: `dict`
            required data schema in a `dict` where `keys` are COLUMN_NAMEs,\r
                and `values` are DATA TYPE.\    
        -----

        '''
        schema = {}
        for col in df.columns:
            copy_df = df[[col]].copy()
            copy_df = copy_df[copy_df.isnull().any(axis=1)==False].head(1)
            schema[col] = 'object'
            if copy_df.shape[0] == 0:
                continue
            schema[col] = str(copy_df.dtypes[0])
            if schema[col] != 'object':
                continue
            if pd.to_numeric(copy_df[col], errors='coerce',
                downcast='integer').notnull().all():
                schema[col] = 'int64'
            elif pd.to_numeric(copy_df[col], errors='coerce',
                downcast='float').notnull().all():
                schema[col] = 'float64'
            elif pd.to_datetime(df[col], errors='coerce'
                                            ).notnull().all():
                schema[col] = 'datetime64'
            
        return schema
    
    def check_data_type(df: pd.DataFrame, 
                        schema:dict) -> pd.DataFrame:
        '''
        `check_data_type` checks if data type of all columns of a dataset are \r
            matched or convertable to required data type.

        Parameters
        ----------
        
        df: `DataFrame`
            input dataset
        schema: `dict`
            required data schema in a `dict` where `keys` are COLUMN_NAMEs,\r
                and `values` are DATA TYPE.\        

        Returns
        -------
        a `DataFrame` showing all columns data type, required type, and if they \r
            are convertable.

        Notes
        - Acceptable data types `['int', 'float', 'datetime','str']`
        -----

        '''
        exp = []
        for col in df.columns:
            is_valid_type = False
            if col not in schema.keys():
                exp.append([col, df.dtypes[col],"Not Defined", False])
                continue
            if schema[col]=='int':
                is_valid_type = pd.to_numeric(df[col], errors='coerce',
                                            downcast='integer').notnull().all()
            elif schema[col]=='float':
                is_valid_type = pd.to_numeric(df[col], errors='coerce',
                                            downcast='float').notnull().all()
            elif schema[col]=='datetime':
                is_valid_type = pd.to_datetime(df[col], errors='coerce'
                                            ).notnull().all()
            else:
                is_valid_type = True
            exp.append([col, df.dtypes[col],schema[col], is_valid_type])
        return pd.DataFrame(exp, columns= ['Column Name', 
                                    'Date Type', 
                                    'Required Type', 
                                    'Is Convertable'])

    def check_range(df : pd.DataFrame, 
                    col_name : str, 
                    l_bound = None, 
                    u_bound = None, 
                    is_datetime :bool = False) -> None:
        '''
        `check_range` checks if a give column is distributed in a logical range

        Parameters
        ----------
        df: `DataFrame`
            input dataset
        col_name: `str`
            column name
        l_bound: numeric (`int`, `float`) or datetime (`str`, `datetime`),\r
            default = `None`
            lower bound
        u_bound: numeric (`int`, `float`) or datetime (`str`, `datetime`),\r
            default = `None`
            upper bound
        is_datetime: `bool`, default = `False`
            a boolean to show if the given column's data type is datetime

        Returns
        -------
        a chart and table showing the data distribution.

        Notes
        - if l_bound is not define, it takes `MEDIAN - 1.5 * IQR` as lower bound
        - if u_bound is not define, it takes `MEDIAN + 1.5 * IQR` as lower bound
        -----

        '''
        fig, ax = plt.subplots(1,2, figsize=(8,3))
        fig.subplots_adjust(wspace=0, top=1, right=1, left=0, bottom=0)
        
        if is_datetime:
            is_na = np.isnat(pd.to_datetime(df[col_name].values))
            new_df = df[is_na==False]
            array_d = new_df[col_name].values.astype('datetime64')
            ax[1].hist(array_d)
            ax[1].set_xlabel(col_name) 
            array = new_df[col_name].values.astype('datetime64[ns]').astype('float')
            iqr = np.nanpercentile(array, 75)-np.nanpercentile(array, 25)
            median = np.nanmedian(array)
            if l_bound==None:
                l_bound = (median-1.5*iqr).round(3)
            else:
                l_bound = np.datetime64(l_bound, 'ns').astype('float')
            if u_bound==None:
                u_bound = (median+1.5*iqr).round(3) 
            else:
                u_bound = np.datetime64(u_bound, 'ns').astype('float')
            is_outlier = (array<l_bound)*1+(array>u_bound)*1
            p_outlier = 100*is_outlier.sum()/len(array)
            sum_df = pd.DataFrame({'features': ['count',
                'mean',
                'median',
                'min, max',
                'LB, UB',
                'outliers %'],
                'values':
                    [len(array),
                    np.datetime_as_string(
                        np.datetime64(int(np.nanmean(array)), "ns"), unit='D'),
                    np.datetime_as_string(
                        np.datetime64(int(np.nanmedian(array)), "ns"), unit='D'),
                    [np.datetime_as_string(
                        np.datetime64(int(np.nanmin(array)), "ns"), unit='D'),
                    np.datetime_as_string(
                        np.datetime64(int(np.nanmax(array)), "ns"), unit='D')],
                    [np.datetime_as_string(
                        np.datetime64(int(l_bound), "ns"), unit='D'),
                    np.datetime_as_string(
                        np.datetime64(int(u_bound), "ns"), unit='D')],
                    p_outlier.round(3)]
            })
            the_table = ax[0].table(cellText=sum_df.values,
                colLabels=sum_df.columns, 
                bbox=[0.0, 0.0, 0.9, 1])
        else:
            is_na = np.isnan(df[col_name].values.astype('float'))
            new_df = df[is_na==False]
            array = new_df[col_name].values.astype('float')
            ax[1].boxplot(array)
            ax[1].set_ylabel(col_name)

            iqr = np.nanpercentile(array, 75)-np.nanpercentile(array, 25)
            median = np.nanmedian(array)
            if l_bound==None:
                l_bound = (median-1.5*iqr).round(3)
                if l_bound< array.min():
                    l_bound = array.min()
            if u_bound==None:
                u_bound = (median+1.5*iqr).round(3)
                if u_bound> array.max():
                    u_bound = array.max()
            is_outlier = (array<l_bound)*1+(array>u_bound)*1
            p_outlier = 100*is_outlier.sum()/len(array)
            sum_df = pd.DataFrame({'features': ['count',
                    'mean',
                    'std',
                    'min, max',
                    'LB, UB',
                    'outliers %'],
                    'values':
                        [len(array),
                        np.nanmean(array).round(3),
                        np.nanstd(array).round(3),
                        [np.nanmin(array).round(2), np.nanmax(array).round(2)],
                        [l_bound, u_bound],
                        p_outlier.round(3)]
                })
            the_table = ax[0].table(cellText=sum_df.values,
                colLabels=sum_df.columns, 
                bbox=[0.0, 0.0, 0.75, 1])
        
        
        ax[0].axis('off')
        ax[0].axis('tight')

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)
        the_table.auto_set_column_width(col=list(range(2)))
        if p_outlier>0:
            print("{} records ({}%) are found out of the given range.".format(
                is_outlier.sum(),
                p_outlier.round(3)))
        plt.show()


    def is_datetime_valid(df: pd.DataFrame,
                        col_name: str,
                        format: 'str' = '%Y-%m-%dT%H:%M:%S.%f000') -> None:
        '''
        `is_datetime_valid` checks if a give column follows a given datatime fomrat

        Parameters
        ----------
        df: `DataFrame`
            input dataset
        col_name: `str`
            column name
        format: `str`, default = `'%Y-%m-%dT%H:%M:%S.%f000'`
            datetime format

        Returns
        -------
        it prints howmany records are not in the given format
        '''
        array= df[col_name].values.astype('str')
        def my_check(x):
            try:
                return bool(datetime.strptime(x, format))
            except ValueError:
                return False

        vec_function = np.vectorize(my_check)
        is_valid = vec_function(array)
        n = len(array)
        n_valid = is_valid.sum()
        if np.all(is_valid):
            print("all {} are in {} format.".format(col_name, format))
        else:
            print("{} records ({}%) of {} are not in {} format.".format(
                (n - n_valid), 100*(n - n_valid)/n, col_name, format
            ))
            
    def check_duplicate_rows(df: pd.DataFrame, 
                            indices: list[str] = None) -> pd.DataFrame:
        
        '''
        `check_duplicate_rows` checks if there is any duplicate rows in the given\r
            dataset

        Parameters
        ----------
        df: `DataFrame`
            input dataset
        indices: `list[str]`, default = `None`
            list of column names considered as index

        Returns
        -------
        duplicated_data: `DataFrame`
            duplicated rows

        Notes
        -----
        - if indices is not defined, it takes a list of all columns as indices.
        '''

        if indices == None:
            indices = df.columns
        duplicated_rows = df[indices].duplicated(keep=False)
        duplicated_data = df[duplicated_rows]
        duplicated_data.sort_values(by = indices,
                                    inplace=True)
        

        print("{} out of {} rows are duplicated".format(
            duplicated_data.shape[0],
            df.shape[0]
        ))

        return duplicated_data

    def is_blank(df: pd.DataFrame,
                col_name: str,
                data_type: str = 'str') -> None:
        '''
        `is_blank` checks if there is any blank in the given column

        Parameters
        ----------
        df: `DataFrame`
            input dataset
        col_name: `str`
            column name
        data_type: `str`, default = `str`
            required data type

        Returns
        -------
        it prints how many records are blank
        '''
        if data_type in ('int', 'float'):
            is_na = np.isnan(df[col_name].values.astype('float'))
        elif data_type=='datetime':
            is_na = np.isnat(pd.to_datetime(df[col_name].values))
        else:
            is_na = (df[col_name].values.astype('str')=="") |\
                    (df[col_name].values.astype('str')=="nan") |\
                    (df[col_name].values.astype('str')=="None")|\
                    (df[col_name].values.astype('str')==" ") |\
                    (df[col_name].values.astype('str')=="NULL")
        if np.any(is_na):
            print("There are {} records ({} %) where {} is not entered.".format(
                is_na.sum(),
                is_na.sum()/len(is_na),
                col_name
            ))
        else:
            print("There is no record where {} is not entered.".format(
                col_name
            ))


# DATA PROFILING
class data_profiling:
    def summarize_data(df, columns = None, n_visible_columns = 10):
        def sum_data(df, col_name, is_date=False, is_cat=False):
            data = df[col_name].values
            n_digit = 2
            n = len(data)
            fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
            fig.subplots_adjust(wspace=0, top=1, right=1, left=0, bottom=0)
            if is_cat:
                in_valid = ["", "nan", "NA", "NULL", "None"]
                data = data.astype("str")
                is_null = np.in1d(data.astype("str"), in_valid)
                data[is_null] = "nan"
                unique_values, value_counts = np.unique(data, return_counts=True)
                index = np.argsort(value_counts)[::-1]
                unique_values, value_counts = (
                    unique_values[index],
                    value_counts[index],
                )

                if len(unique_values) > n_visible_columns:
                    unique_values = np.append(unique_values[:n_visible_columns], "other")
                    value_counts = np.append(
                        value_counts[:n_visible_columns], value_counts[n_visible_columns:].sum()
                    )
                sum_df = {
                    "Size": n,
                    "Distinct": len(np.unique(data)),
                    "Missing %": round(100 * is_null.sum() / n, n_digit),
                }
                for i, val in enumerate(unique_values[:5]):
                    sum_df["{} %".format(val)] = round(100*value_counts[i]/n, n_digit)

                ax[1].barh(unique_values, value_counts)
                ax[1].set_ylabel(col_name)
            elif is_date:
                data_num = data.astype("int64")
                sum_df = {
                    "Size": n,
                    "Distinct": len(np.unique(data)),
                    "Missing %": round(100 * np.isnat(data).sum() / n, n_digit),
                    "Zeros %": round(100 * (data_num == 0).sum() / n, n_digit),
                    "Mean": np.datetime_as_string(
                        np.datetime64(int(np.nanmean(data_num)), "ns"), "D"
                    ),
                    "Median": np.datetime_as_string(
                        np.datetime64(int(np.nanmedian(data_num)), "ns"), "D"
                    ),
                    "Range": [
                        np.datetime_as_string(np.datetime64(int(np.nanmin(data)), "ns"), "D"),
                        np.datetime_as_string(np.datetime64(int(np.nanmax(data)), "ns"), "D"),
                    ],
                    "IQR": [
                        np.datetime_as_string(
                            np.datetime64(int(np.nanpercentile(data_num, 25)), "ns"), "D"
                        ),
                        np.datetime_as_string(
                            np.datetime64(int(np.nanpercentile(data_num, 75)), "ns"), "D"
                        ),
                    ],
                    "95 % CI": [
                        np.datetime_as_string(
                            np.datetime64(int(np.nanpercentile(data_num, 2.5)), "ns"), "D"
                        ),
                        np.datetime_as_string(
                            np.datetime64(int(np.nanpercentile(data_num, 97.5)), "ns"), "D"
                        ),
                    ],
                }
                ax[1].hist(data, 20, edgecolor="white")
                ax[1].set_xlabel(col_name)
            else:
                sum_df = {
                    "Size": n,
                    "Distinct": len(np.unique(data)),
                    "Missing %": round(100 * np.isnan(data).sum() / n, n_digit),
                    "Zeros %": round(100 * (data.astype("float") == 0).sum() / n, n_digit),
                    "Mean": round(np.nanmean(data), n_digit),
                    "Median": round(np.nanmedian(data), n_digit),
                    "St. Dev.": round(np.nanstd(data), n_digit),
                    "Range": [round(np.nanmin(data), n_digit), round(np.nanmax(data), n_digit)],
                    "IQR": [
                        round(np.nanpercentile(data, 25), n_digit),
                        round(np.nanpercentile(data, 75), n_digit),
                    ],
                    "95 % CI": [
                        round(np.nanpercentile(data, 2.5), n_digit),
                        round(np.nanpercentile(data, 97.5), n_digit),
                    ],
                }
                ax[1].hist(data, 20, edgecolor="white")
                ax[1].set_xlabel(col_name)

            the_table = ax[0].table(
                cellText=list(sum_df.items()),
                colLabels=["metric", "value"],
                bbox=[0.0, 0.0, 0.75, 1],
            )
            the_table.auto_set_font_size(False)
            the_table.set_fontsize(9)
            the_table.auto_set_column_width(col=[0])
            # Hide the box border
            for key, cell in the_table._cells.items():
                cell.set_linestyle('-')
                cell.set_edgecolor('gray') 
                cell.set_linewidth(0.5)
            ax[0].axis("off")
            ax[0].axis("tight")
            plt.gca().spines["top"].set_visible(False)
            plt.gca().spines["right"].set_visible(False)
            plt.show()

        df = df.copy()
        if columns==None:
            columns= df.columns
        datetime_columns = df.select_dtypes(include=["datetime64", "<M8"]).columns.tolist()
        categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

        for col_name in columns:
            if df[col_name].dtype not in categorical_columns and\
                len(df[col_name].unique())<10:
                categorical_columns.append(col_name)
                df[col_name] = df[col_name].astype('category')
            data = df[col_name].values
            sum_data(
                df,
                col_name,
                is_cat=(col_name in categorical_columns),
                is_date=(col_name in datetime_columns),
            )

    def xy_correlation(df, x_col, y_col, n_visible_columns = 20):
        col_type = df[x_col].dtype
        data = df[[y_col,x_col]].copy()
        if col_type == 'object':
            data[x_col] = data[x_col].values.astype('str')
            sum_df = data.groupby(
                by =[x_col]).agg({
                    y_col:'max'
                }).reset_index()

            sum_df.sort_values(
                by = y_col, 
                ascending=False,
                inplace=True)

            labels = []
            for i in range(len(sum_df.index)):
                lb = "other"
                if i<20:
                    lb = str(sum_df.iloc[i,0])[0:20]
                labels.append(lb)

            sum_df['label'] = labels
            sum_df.drop(y_col, axis=1, inplace=True)
            data = data.merge(sum_df, on = [x_col])

            data[x_col] = pd.Series(
                data['label'].values)

            ax = data.boxplot(
                by=x_col,
                grid=False, 
                rot=90, 
                fontsize=10)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.title('')
            plt.suptitle('')
            plt.show
        else:
            ax = df.plot.scatter(x=x_col,
                y=y_col,
                grid=False, 
                rot=90, 
                fontsize=10)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            plt.title('')
            plt.suptitle('')
            plt.show


class stats_analysis:
    def t_test(sample1, sample2, title1, title2, plot_title = '', x_label ='',
               y_label='frequency', one_tail = False, add_curve = False):
        '''
        TEST
        '''
        test_type = 'two-sided'
        h0 = 'H0: mu({0} - {1}) = 0'.format(title1, title2) 
        ha = 'H1: mu({0} - {1}) <> 0'.format(title1, title2)
        message = '{0} and {1} are significantly different'.format(title1, title2)

        if one_tail:
            test_type = 'two-sided'
            h0 = 'H0: mu({0} - {1}) = 0'.format(title1, title2) 
            ha = 'H1: mu({0} - {1}) > 0'.format(title1, title2)
            message = '{0} is significantly greater than {1}'.format(title1, 
                                                                    title2)
            
            if sample1.mean() < sample2.mean():
                message = '{0} is significantly greater than {1}'.format(title2, 
                                                                    title1)
        res = stats.ttest_ind(sample1, sample2, alternative= test_type,
                            equal_var = True)

        print("**T-test for two samples with equal variences:**\n")
        print(h0 + '\n' + ha + '\n\n')


        print("P-Value = {:.3f}".format(res.pvalue))
        if res.pvalue < 0.05:
            print("we can REJECT H0: " + message)
        else:
            print("we CANNOT reject H0")


        fig, ax = plt.subplots()

        sns.histplot(x=sample1, ax=ax, kde=add_curve, linewidth =0, 
                     color='navy', label=title1, alpha=0.6)
        sns.histplot(x=sample2, ax=ax, kde=add_curve, linewidth =0, 
                     color='red', label=title2, alpha=0.6)
        ax.set_title(plot_title)
        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        plt.legend()
        plt.show()
    def one_way_anova(df,feature, col_name, title):
        df.columns = df.columns.str.replace(" ", "_")
        col_name = col_name.replace(" ", "_")
        feature = feature.replace(" ", "_")

        df.boxplot(by=col_name, column=feature)
        plt.xticks(rotation=90, ha='right')
        plt.title(title)
        plt.suptitle('')
        plt.ylabel(feature)
        plt.grid(None)


        model = ols('{} ~ C({})'.format(feature, col_name), 
                        data=df).fit()

        anova_table = sm.stats.anova_lm(model, typ=2)

        print("\n**ONE-WAY ANOVA:**\n")
        print("H0: mu1 = mu2 = mu3 = ... \
            \nHa: at least one is different\n\n")

        print(anova_table)

        print("\n\n")
        p_value = anova_table.iloc[0]['PR(>F)']
        if p_value < 0.05:
            print("we can REJECT H0: {} significantly affects {}".format(
                col_name,feature
            ))
        else:
            print("we CANNOT reject H0: there is no difference")
        plt.show()

    def normality_test(array: np.ndarray, 
        alpha: float = 0.05, 
        show_plots: bool = True, 
        label: str = 'value') -> pd.DataFrame:
        '''
            `normality_test` runs multiple normailty tests on an array

            Parameters
            ----------
            array: `np.ndarray`
                input array
            alpha: `decimal`
                alpha value, default = 0.05
            show_plots: `bool`
                a flag to show or hide plots, default = True
            label: `str`
                x label of plot, default = value

            Returns
            -------
            a table showing Normailty tests.

            Notes        
            -----
                - it's recommended to keep the sample size lower than 5000.
                - for shapiro, normaltest, kstest, we calculate p-value and compare
                    it with alpha  (0.05):
                    - p <= alpha: reject H0, not normal.
                    - p > alpha: fail to reject H0, normal.
        '''
        
        if show_plots:
            fig , axs = plt.subplots(1,2, figsize = (7,3))
            sns.histplot(x=array, ax=axs[0], kde=True, linewidth =0,
                    color='navy')
            qqplot(array, line='s', ax = axs[1])
            axs[0].set_xlabel(label)
            axs[0].set_title('Histogram')
            axs[1].set_title('QQ plot')
            plt.show()

        res = []
        ci = '{:.2f}%'.format(1 - alpha)
        for method in [shapiro, normaltest, kstest]:
            stat, p = shapiro(array)
            result = 'Normal' if p > alpha else 'NOT Normal'

            res.append([method.__name__, stat, p, ci, result])

        ad_results = anderson(array)
        stat = ad_results.statistic
        for i in range(len(ad_results.critical_values)):
            f = ad_results.critical_values[i]
            sl = ad_results.significance_level [i]/100
            ci = '{:.2f}%'.format(1 - sl)
            result = 'Normal' if stat < f else 'NOT Normal'
            res.append([anderson.__name__, stat, 'NA', ci, result])

        res = pd.DataFrame(res, 
            columns = ['Test Name', 'Stat', 'P-value', 'CI','Result'])

        def cond_formatting(val):
            if val == 'NOT Normal':
                return 'background-color: red'
            if val == 'Normal':
                return 'background-color: lightgreen'
            
            return ''

        res = res.style.applymap(cond_formatting)
        return res

class data_exploration:
    def plot_pareto(df : pd.DataFrame, 
                    col_name : str,
                    top_k : int = 10, 
                    show_cum_percentage : bool = True) -> None:
        '''
            `plot_pareto` plots pareto on a give column of a dataframe

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            col_name: `str`
                column name
            top_k: numeric (`int`), default = `10`
                showing only top k most frequenct groups and combining the rest into 'other'
            show_cum_percentage: `bool`, default = `True`
                a boolean to show if user wants to add cumulative percentage on plot.

            Returns
            -------
            a pareto chart.

            Notes        
            -----

            '''
        sum_df = df.groupby(by = [col_name]).size().reset_index(name ='counts')
        sum_df.sort_values(by = 'counts', ascending = False, inplace = True)
        if len(sum_df)>top_k:
            other = sum_df.iloc[top_k:,1].sum()
            sum_df = sum_df[:top_k]
            sum_df.loc[top_k] = ['other', other] 
        else:
            top_k = (1+len(sum_df)//5)*5
            for i in range(len(sum_df),top_k):
                sum_df.loc[i] = ['', 0]

        labels = [str(lb) for lb in sum_df[col_name].values]
        sum_df[col_name] = labels
        percents = sum_df['counts'].values
        percents = percents/percents.sum()
        sum_df['percentage'] = np.around(percents, decimals=2)
        sum_df["cumpercentage"] = sum_df["counts"].cumsum()/sum_df["counts"].sum()


        fig, ax = plt.subplots()
        ax.bar(sum_df[col_name], sum_df["counts"])
        
        ax.set_xlabel(col_name)
        ax.set_ylabel('frequency')
        ax.tick_params(axis="y")

        if show_cum_percentage:
            ax2 = ax.twinx()
            ax2.plot(sum_df[col_name], sum_df["cumpercentage"], color="gray", 
                    marker="o", ms=5)
            ax2.set_ylabel('cum. percentage')
            ax2.tick_params(axis="y", colors="gray")
        plt.show()

    def plot_histogram(df : pd.DataFrame,
                        col_name : str, 
                        bins : int = 50) -> None:
        '''
            `plot_histogram` plots histogram on a give column of a dataframe

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            col_name: `str`
                column name
            bins: numeric (`int`), default = `50`
                number of bins.

            Returns
            -------
            a histogram.

            Notes        
            -----

            '''
        array = df[col_name].values

        fig, axs = plt.subplots(2, 
            sharex=True, 
            gridspec_kw={"height_ratios": (.15, .85)})

        sns.boxplot(x=array, ax=axs[0])
        sns.histplot(x=array, ax=axs[1], bins= bins, kde=True, linewidth =0,
            color='navy')
        axs[0].set(yticks=[])
        axs[1].set_xlabel(col_name)
        axs[1].set_ylabel("frequency")
        sns.despine(ax=axs[0], left=True)
        sns.despine(ax=axs[1])
        plt.show()

    def plot_3d(df : pd.DataFrame, 
                x_col : str, 
                y_col : str,
                z_col : str) -> None:
        '''
            `plot_3d` plots a 3D scatter plot taking three columns of a dataset 

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            x_col: `str`
                x column name
            y_col: `str`
                y column name
            z_col: `str`
                z column name

            Returns
            -------
            a 3D scatter plot.

            Notes        
            -----

            '''
        plt.style.use('default')
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        ax.scatter(df[x_col].values, df[y_col].values, df[z_col].values)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        plt.show()

    def plot_concentration_map (df: pd.DataFrame, 
        label_col: str, 
        val_col: str, 
        lat_col: str = 'LATITUDE', 
        lon_col: str = 'LONGITUDE', 
        colormap: object = None) -> object:

        '''
            `plot_concentration_map` plots points on a map 

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            label_col: `str`
                label column name
            val_col: `str`
                value column name
            lat_col: `str`
                latitude column name, default = `LATITUDE`
            lon_col: `str`
                longitude column name, default = `LONGITUDE`
            colormap: `branca.colormap`
                colormap of branca package, default = None
            
            Returns
            -------
            a folium map.

            Notes        
            -----
                if colormap is None it creates a Blue to Red Colormap by default.
            '''

        r = df[val_col].values
        if colormap == None:
            colormap = cm.linear.RdBu_09.scale(r.min(), r.max())
            colormap.caption = val_col
        r = 50000*(r - r.min() + 0.0001)/(r.max() - r.min() + 0.0001)
        df['radius'] = np.array(r).astype('int')
        mymap = folium.Map(width =700, height = 420, location = [36,-99], 
            zoom_start = 4)
            

        for i, row in df.iterrows():
            folium.Circle(location=[row[lat_col], row[lon_col]], 
                popup = "{}: {:.0f}\r{}: {:.3f}".format(label_col, row[label_col],
                    val_col, row[val_col]), 
                fill_color = colormap(row[val_col]),
                fill_opacity = 0.9, 
                radius = row['radius'], 
                weight = 1, 
                color = '#000').add_to(mymap)
                
        mymap.add_child(colormap)

        return mymap

    def xy_scatterplot(df: pd.DataFrame, 
        x_col: str, 
        y_col: str) -> None:
        '''
            `xy_scatterplot` shows x and y on a scatterplot

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            x_col: `str`
                x column name
            y_col: `str`
                y column name
            
            Returns
            -------
            a scatter plot.

            Notes        
            -----
                x, and y should be continuous variables.
            '''
        ax = df.plot.scatter(x=x_col,
            y=y_col,
            grid=False, 
            rot=90, 
            fontsize=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.title('')
        plt.suptitle('')
        plt.show()

    def xy_boxplot(df: pd.DataFrame, 
        x_col: str, 
        y_col: str,
        top_k = 20) -> None:
        '''
            `xy_boxplot` shows x and y on a group boxplot

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            x_col: `str`
                x column name
            y_col: `str`
                y column name
            top_k:  `int`
                maximum number of gourps, default = 20

            Returns
            -------
            a group box plot.

            Notes        
            -----
                - y should be continuous variable.
                - x should be categorical variable.
            '''
        data = df[[y_col,x_col]].copy()
        data[x_col] = data[x_col].values.astype('str')
        sum_df = data.groupby(
            by =[x_col]).agg({
                y_col:'max'
            }).reset_index()

        sum_df.sort_values(
            by = y_col, 
            ascending=False,
            inplace=True)

        labels = []
        for i in range(len(sum_df.index)):
            lb = "other"
            if i<top_k:
                lb = str(sum_df.iloc[i,0])[0:top_k]
            labels.append(lb)

        sum_df['label'] = labels
        sum_df.drop(y_col, axis=1, inplace=True)
        data = data.merge(sum_df, on = [x_col])

        data[x_col] = pd.Series(
            data['label'].values)

        ax = data.boxplot(
            by=x_col,
            grid=False, 
            rot=90, 
            fontsize=10)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.title('')
        plt.suptitle('')
        plt.show()

    def plot_correlation(df: pd.DataFrame, 
        cols: list[str]) -> None:
        '''
            `plot_correlation` plots correlation matrix between given columns

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            cols: `list[str]`
                list of column names

            Returns
            -------
            a correlation plot.

            Notes        
            -----
                - columns should be continuous variable.
                - it skips categorical columns by default.
            '''

        sns.set_theme(style='white')
        sns.pairplot(df[cols], height = 2.5)
        plt.show()

    def plot_timeseries(df: pd.DataFrame, 
        time_col: str, 
        val_cols: list[str]) -> None:
        '''
            `plot_timeseries` plots given column(s) over the time

            Parameters
            ----------
            df: `DataFrame`
                input dataset
            time_col: `str`
                date column name
            val_cols: `list[str]`
                list of column names

            Returns
            -------
            a timeserie plot.

            Notes        
            -----
                - columns should be continuous variable.
        '''
        clrs = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig, ax = plt.subplots(len(val_cols), 1, sharex=True)
        if len(val_cols)==1:
            ax = [ax]
        fig.subplots_adjust(hspace=0)
        
        for i, col in enumerate(val_cols):
            df.plot(x = time_col, y = col, ax= ax[i], color = clrs[i])
            ax[i].legend(loc='upper left', prop={'size':10})

        plt.show()
