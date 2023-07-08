import pyarrow.parquet as pq
import awswrangler as wr
import pandas as pd
import numpy as np

from evidently.test_suite import TestSuite
from evidently.tests import *

import traceback
import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

__all__ = ["DriftReport"]


class ResultInstance:
    """The ResultInstance class is used to store all drift scores for all periods provided
    or found in the data. Additionally it handles the plotting
    

    Parameters
    ----------

    Attributes
    ----------
    plot : 
        Plots the data distribution for each variable in the set of features
    save : 
        Saves the data distribution for each variable in the set of features into a pdf file
    """
    
    def __init__(self):
        self.drift_scores = {}
        self.mode = None
        self.name = 'Drift Report'
        self.periods = []
        self.feat = []
        self.method = None
        self.map_method = {'jensenshannon':'Jensen-Shannon distance', 'kl_div':'Kullback-Leibler divergence',
                           'wasserstein':'Wasserstein distance (normed)', 'hellinger':'Hellinger Distance (normed)'}
        
    
    def parse(self):
        if self.periods:
            parsed = pd.DataFrame()
            for p in self.periods[1:]:
                tmp = pd.DataFrame.from_dict(self.drift_scores[str(p)]['tests'][0]['parameters']['features'])
                tmp = tmp.reset_index().rename(columns={'index': 'info'})
                tmp['codmes'] = p
        
                parsed = pd.concat([parsed, tmp])
            parsed = parsed.reset_index(drop=True)
            parsed['codmes'] = parsed['codmes'].astype('string')
            
        return parsed
    
    def plot(self):
        parsed = self.parse()
        filtered = parsed[parsed['info']=='score']

        nrow = int(len(self.feat)/2) if len(self.feat)%2 == 0 else int(len(self.feat)/2) + 1
        
        fig, axes = plt.subplots(nrow, 2, figsize=(18, round(6.66*nrow)))
        for i in range(len(self.feat)):
            #print(int(i/2),int(i%2))
            sns.lineplot(ax=axes[int(i/2), i%2],
                data=filtered,
                x="codmes", y=self.feat[i], hue="info", style='info',
                markers=True, dashes=False
            )
            axes[int(i/2), i%2].axhline(0.2, ls='-', c='green', linewidth=0.5, alpha=.7)
            axes[int(i/2), i%2].set_title(self.feat[i])

        fig.suptitle(self.name, x=0.3, y=0.93, fontsize=24, weight='bold', c='#1eba20')
        fig.text(x=0.19, y=0.903, s=self.mode, fontsize=15, weight='bold')
        fig.text(x=0.255, y=0.903, s=self.map_method[self.method], fontsize=15, fontstyle='italic', c='#cecbcb', weight='normal')
        fig.text(x=0.50, y=0.903, s='Last codmes: '+str(self.periods[-1]), fontsize=15, fontstyle='italic', weight='regular')
        
        fig1 = plt.gcf()
        #fig1.savefig("test.pdf", format='pdf')
        #return fig1


    
    def save(self, namefile='driftReport.pdf'):
        parsed = self.parse()
        filtered = parsed[parsed['info']=='score']
        
        
        with PdfPages(namefile) as pdf:

            npages = len(self.feat)//6 if len(self.feat)%6 == 0 else 1 + len(self.feat)//6
            for pg in range(npages):
                var = self.feat[pg*6:pg*6+6]
                nrow = len(var)//2 if len(var)%2 == 0 else 1 + len(var)//2

                #fig, axes = plt.subplots(nrow, 2, figsize=(18, round(6.6*nrow))) # less empty boxes
                fig, axes = plt.subplots(3, 2, figsize=(18, 20))
                for i in range(len(var)):
                    sns.lineplot(ax=axes[int(i/2), i%2],
                        data=filtered,
                        x="codmes", y=var[i], hue="info", style='info',
                        markers=True, dashes=False
                    )
                    axes[int(i/2), i%2].axhline(0.2, ls='-', c='green', linewidth=0.5, alpha=.7)
                    axes[int(i/2), i%2].set_title(var[i])

                if pg == 0:
                    fig.suptitle(self.name, x=0.3, y=0.93, fontsize=24, weight='bold', c='#1eba20')
                    fig.text(x=0.19, y=0.903, s=self.mode, fontsize=15, weight='bold')
                    fig.text(x=0.255, y=0.903, s=self.map_method[self.method], fontsize=15, fontstyle='italic', c='#cecbcb', weight='normal')
                    fig.text(x=0.50, y=0.903, s='Last codmes: '+str(self.periods[-1]), fontsize=15, fontstyle='italic', weight='regular')

                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

        return "Report saved to: " + namefile
    
    
class DataInstance:
    """The DataInstance class is used to manipulate and validate data quality for 
    the analysis.
    

    Parameters
    ----------
    features : List[str]
        List of features
    dtype_map : dict
        Mapping of database types into pandas datatype, it's important to verify and provide
        the correct mapping or the scores might be not correct.
        Example:
        DB type | pandas dtype
        'string': 'object'

    Attributes
    ----------
    set_reference : pd.DataFrame
        sets the reference or base data for the drift computations
        
    set_current : pd.DataFrame
        sets the current data for the drift computations with respect to reference
    """
    
    def __init__(self, features: list['str'] | None, 
                 dtype_map: dict = {'string':'object', 'category':'object', 'Int64':'int64'}) -> None:
        self.features = features
        self.dtype_map = dtype_map
        self.reference = None
        self.current = None
        self.valid_ref = False
        self.valid_cur = False
        
    def set_reference(self, ref: pd.DataFrame | None):
        if self.valid_columns(ref):
            self.valid_ref = True
        
            self.reference = self.verify_dtypes(ref)
        
        else:
            self.valid_ref = False
            
        
    def set_current(self, cur: pd.DataFrame | None):
        if self.valid_columns(cur):
            self.valid_cur = True
        
            self.current = self.verify_dtypes(cur)
        else:
            self.valid_cur = False

        
    def verify_dtypes(self, data):
        data_types = data[self.features].dtypes.apply(lambda x: x.name).to_dict()

        for f in self.features:
            if data_types[f] in self.dtype_map:
                data_types[f] = self.dtype_map[data_types[f]]

        mdata = data[self.features].astype(data_types)
            
        return mdata
    
    def valid_columns(self, data):
        inner_feat = set(list(data.columns))
        client_feat = set(self.features)

        valid_cols = False
        if client_feat.issubset(inner_feat):
            valid_cols = True
            
        return valid_cols



class DriftReport:
    """The basic DifritReport class for creating reports of changes in data distribution.
    This class defines a basic class for DriftReport reports.
    The following steps will be executed automatically:

    Parameters
    ----------
    name : str
        Name of the report
    features : List[str]
        A list of features to analyse the drift.
    periods : list
        A list of periods to analyse the drift, if provided, it should be at least two periods.
    mode : str
        Mode analsyis, it can be: `absolute` or `relative`. While the absoulte mode uses a fixed reference, the 
        relative uses variable references.
        Default: absolute
    stats : str
        The statistical metric to compute the distance. Options:'jensenshannon', 'kl_div', 'wasserstein', 'hellinger'
        Wasserstein metric works only with numerical features(all features should be numeric or float)
        Default: jensenshannon
    stats_th : float
        The threshold to detect drift in the distribution
        Default: 0.2

    Attributes
    ----------
    run : **args
        The main attribute to generate the report
    """
    
    def __init__(self, name: str, features: list['str'] | None, periods: list | None, 
                 mode: str = 'absolute', stats: str = 'jensenshannon', stats_th: float = 0.2) -> None:
        self.name = name
        self.features = features
        self.periods = periods    # will be overwritten if None
        self.mode = mode
        self.stats = stats
        self.stats_th = stats_th

    def add_period(self, period):
        self.periods.append(period)

    def is_valid(self, colperiod_name, table_name, db_name):
        valid = False
        if table_name == None and self.periods:
            if len(self.periods) >= 2:
                valid = True
        if table_name:
            if db_name == None:
                valid = False
                print("Se necesita el nombre de la base de datos `db_name`")
            else:
                failed_flag = 0 
                try:
                    print("Detectando periodos...")
                    value = wr.athena.read_sql_query(f"select distinct({colperiod_name}) as codmes from {table_name}", 
                                                     database=db_name)
                except Exception as err:
                    failed_flag = 1
                    print(traceback.format_exc())
                    print(f"Unexpected {err=}, {type(err)=}")
                    print("\nOcurrio un error mientras se detectaban los periodos en la DB")
                    
                if failed_flag == 1:
                    valid = False
                else:
                    if value.empty:
                        print(f'\nLa tabla {db_name}.{table_name} esta vacia')
                        valid = False
                    else:
                        value['codmes'] = value['codmes'].astype('int')
                        self.periods = sorted(list(value['codmes']))
                        print(self.periods)
                        if len(self.periods) >= 2:
                            valid = True
                        else:
                            valid = False
                            print(f'\nNo se tiene suficientes periodos para el analisis')
                            
        return valid
    
        
    def load(self, db_parquet, colpart_name, period, colpo):
        part = f'{colpart_name}={period}/'
        path = os.path.join(db_parquet, part)
        if colpo:
            data = pq.read_table(path, filters=[(colpo,'in',['1'])]).to_pandas(strings_to_categorical=True)
        else:
            data = pq.read_table(path).to_pandas(strings_to_categorical=True)
            
        return data
        
        
    def generate(self, db_parquet, colpart_name, colpo):
        months = sorted(self.periods)
        
        print(f'Analizando cambio de distribucion {self.mode}...')
        data_def = DataInstance(self.features)
        results = ResultInstance()
        for i, m in enumerate(months):
            if i == 0:
                ref = self.load(db_parquet, colpart_name, m, colpo)
                data_def.set_reference(ref)
            if i >= 1:
                cur = self.load(db_parquet, colpart_name, m, colpo)
                data_def.set_current(cur)

                if data_def.valid_ref and data_def.valid_cur:
                    suite = TestSuite(tests=[
                        TestNumberOfDriftedColumns(stattest=self.stats, stattest_threshold=self.stats_th),  
                    ])

                    suite.run(reference_data=data_def.reference, current_data=data_def.current)
                    results.drift_scores[str(m)] = suite.as_dict()
                    results.mode = self.mode
                    results.feat = self.features
                    results.name = self.name
                    results.periods = self.periods
                    results.method = self.stats

                    if self.mode == 'relative':
                        data_def.set_reference(cur)
                
                        
        return results
                        
        
    def run(self, db_parquet: str, table_name: str | None, db_name: str | None, 
            colpo: str | None, colperiod_name: str = 'codmes', colpart_name: str = 'p_codmes'):
        """Checks, validate, process data and generates the report.

        Parameters
        ----------
        db_parquet : Data base parquet path 
            Default: s3://interbank-datalake-dev-us-east-1-058528764918-stage//../tablename/
        table_name : str
            The name of the table where the data is stored
            and it should be asociated with the parquet path(example 'ejecucion')
        db_name : str
            The name of the database where the data is stored(example 'd_perm_aws')
        colpo: str, optional
            The name of the PO column 
            Default: None
        colperiod_name: str
            The name of the column that contains the period of the data (usually `codmes` or `periodo`)
            Default: 'codmes'
        colpart_name: str
            The name of the partition column (usually `p_codmes` or `p_periodo`)
            Default: 'p_codmes'
        """
        
        if not self.is_valid(colperiod_name, table_name, db_name):
            print('Hay algun problema con la base de datos, lectura de la base o los periodos')
            res = None
        else:
            res = self.generate(db_parquet, colpart_name, colpo)
            print('Analysis done!')
            
            
            
        return res
            
        
        
        