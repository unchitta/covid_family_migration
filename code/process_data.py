# %%

import pandas as pd
import numpy as np
import geopandas as gpd
import scipy


# %%

def parental_family_puma(pums):
    """
    This function takes a PUMS dataframe and selects households that 
    satisfy the "extended family criteria" below, then groups data by PUMA and returns
    a count and a percentage of these extended families out of all households in each PUMA
    
    Extended family criteria:
        1. Family households (HHTYPE in [1,2,3]) in which
            a) the head of household or spouse (if married and spouse is present) is at least 50 y/o, or
        2. Non-family households (HHTYPE in [4,5,6,7]) where
            a) householder is at least 50 y/o, and
            b) is either married but no spouse present, separated, divorced, or widowed
            
    Essentially, the criteria selects Baby Boomer and some Gen X families.
    """
    
    criteria_1a_2ab = (~pums['HHTYPE'].isin([0,9])) \
                    & (pums['RELATE'].isin([1,2])) \
                    & (pums['AGE'] >= 50) & (pums['MARST']!=6)

    selected = pums[criteria_1a_2ab]
    selected = selected.drop_duplicates('SERIAL').groupby('PUMA', as_index=False)[['HHWT']].agg('sum')
    selected.rename(columns={'HHWT':'selected_families'}, inplace=True)

    tot_hh_counts = pums.drop_duplicates('SERIAL').groupby('PUMA', as_index=False)[['HHWT']].agg('sum')
    tot_hh_counts.rename(columns={'HHWT':'total_households'}, inplace=True)

    merged = pd.merge(selected, tot_hh_counts, on='PUMA')
    merged['sel_fam_pct'] = np.round(merged['selected_families'] / merged['total_households'], 3)
    
    return merged


def calc_parental_family_proxy_cbsa():


    def clean_puma(puma_code):
        # assumes puma_code is a string
        if len(puma_code) == 3: return "00" + puma_code
        elif len(puma_code) == 4: return "0" + puma_code
        else: return puma_code


    def clean_statefip(statefip):
        # assumes statefip is a string
        return "0" + statefip if len(statefip) == 1 else statefip


    # read ipums data
    pums = pd.read_csv('../data/ipums_2019_family.csv', dtype={'PUMA':str,'STATEFIP':str})
    # make sure PUMAs are 5-digit and state fips are 2-digit
    # combine statefip + puma to get unique PUMAs across all states
    pums['PUMA'] = pums['PUMA'].apply(lambda x: clean_puma(x))
    pums['STATEFIP'] = pums['STATEFIP'].apply(lambda x: clean_statefip(x))
    pums['PUMA'] = pums['STATEFIP'] + pums['PUMA']
    # Select families that satisfy the criteria and
    # calculate total number of selected households in each PUMA
    pumas = parental_family_puma(pums)

    ## Load PUMAS to CBSA crosswalk
    crosswalk = pd.read_csv('../data/geocorr2022_puma12-to-cbsa20.csv', encoding='latin-1',
        dtype={'state':str,'puma12':str,'cbsa20':int}, skiprows=[1], 
        usecols=['state','puma12','cbsa20','afact','pop20'])

    crosswalk['puma12'] = crosswalk['state'] + crosswalk['puma12'] 
    crosswalk.drop(columns=['state'], inplace=True)
    crosswalk = crosswalk.query("cbsa20 != 99999")

    ## Allocate PUMA household statistics to CBSA using afact allocation factor
    df = pd.merge(crosswalk, pumas, how='left', left_on='puma12', right_on='PUMA')
    df['selected_families'] = df['afact'] * df['selected_families']
    df['total_households'] = df['afact'] * df['total_households']
    df = df.groupby('cbsa20').agg(
        selected_families=('selected_families','sum'),
        total_households=('total_households','sum'))
    df = df.round().astype(int)
    df['v'] = df['selected_families'] / df['total_households']

    

    return df.dropna()



def get_phi(v_df):
    phi = pd.read_csv('../data/phi.csv', index_col=0)
    phi = phi.merge(v_df[['v']], left_index=True, right_index=True) # assume df containing v has cbsa code as index
    return phi



def make_ses_data():

    def clean_geoid(geoid):
        return int(geoid[-5:])


    def read_acs_table(table):
        tab = pd.read_csv(f'../data/ACSDT5Y2019.{table}-Data.csv', encoding='latin-1', skiprows=[1])
        tab['geoid'] = tab['GEO_ID'].apply(clean_geoid)
        return tab.set_index('geoid')
    

    tables = {'inc':'B07011', 'units':'B25001', 'sfh':'B25032', 'housing':'B25077'}

    ## pop & pop denisty
    pop = gpd.read_file('../data/acs2019_5yr_B01003_31000US29300.shp')
    pop = pop.to_crs(5070)
    pop['area'] = pop['geometry'].area
    pop['pop_density'] = pop['B01003001'] / (pop['area']/2589988.1103) # population per sq.km.
    pop19 = pop['B01003001'].copy()
    pop['log_pop'] = np.log10(pop['B01003001'])
    pop['log_pop_den'] = np.log10(pop['pop_density'])
    pop['ln_pop'] = np.log(pop['B01003001'])
    pop['ln_pop_den'] = np.log(pop['pop_density'])
    pop['geoid'] = pop['geoid'].apply(lambda x: int(x[-5:]))
    pop = pop.set_index('geoid')[['log_pop','log_pop_den','ln_pop','ln_pop_den']]
    
    ## median income
    inc = read_acs_table(tables['inc'])
    inc['ln_median_income'] = np.log(inc['B07011_001E'])
    inc = inc[['ln_median_income']]

    ## housing units
    units = read_acs_table(tables['units'])
    units['ln_housing_units'] = np.log(units['B25001_001E'])
    units = units[['ln_housing_units']]

    ## sfh pct
    sfh = read_acs_table(tables['sfh'])
    sfh['sfh_pct'] = (sfh['B25032_003E'] + sfh['B25032_014E']) / sfh['B25032_001E'] # pct of single family homes out of occupied housing units
    sfh = sfh[['sfh_pct']]

    ## housing median value
    housing = read_acs_table(tables['housing'])
    housing['ln_median_housing_value'] = np.log(housing['B25077_001E'])
    housing = housing[['ln_median_housing_value']]

    ## employment total (agg from BLS county level)
    emp = pd.read_csv('../data/bea_county_total_employment.csv', dtype={'GeoFips':str, '2019':float, '2020':float}, na_values='(NA)')
    county_cbsa_cw = pd.read_csv('/Users/unchitta/Research/data/census/clean/cbsa_county_delineation_mar_2020.csv', dtype={'county_fips':str})
    emp = emp.merge(county_cbsa_cw.set_index('county_fips')[['cbsa_code']], left_on='GeoFips', right_index=True)
    emp = emp.groupby('cbsa_code', as_index=False).agg(emp_19 = ('2019','sum'), emp_20 = ('2020','sum'))
    emp['geoid'] = emp['cbsa_code']
    emp['jobs_per_person_2019'] = emp['emp_19'] / pop19
    emp['jobs_per_person_2020'] = emp['emp_20'] / pop19
    emp['ln_emp_19'] = np.log(emp['emp_19'])
    emp['ln_emp_20'] = np.log(emp['emp_20'])
    emp['ln_jobs_per_person_19'] = np.log(emp['jobs_per_person_2019'])
    emp['ln_jobs_per_person_20'] = np.log(emp['jobs_per_person_2020'])
    emp = emp.set_index('geoid')[['ln_emp_19','ln_emp_20','ln_jobs_per_person_19','ln_jobs_per_person_20']]

    ## combine controls into 1 df
    dfs = [pop, inc, units, sfh, housing, emp]
    cbsa_controls = dfs[0].join(dfs[1:])


    return cbsa_controls



def make_twfe_table(master):
    # make table for running TWFE (DiD) model
    # time period is separated into pre-covid shock and post-coivd shock

    master['ln_pre_ratio'] = np.log(master['pre_ratio'])
    master['ln_post_ratio'] = np.log(master['post_ratio'])
    df1 = master[['cbsa','ln_pre_ratio','v','ln_pop','ln_pop_den','ln_median_income',
                'ln_housing_units', 'sfh_pct', 'ln_median_housing_value','ln_jobs_per_person_19']].copy()
    df1['CBSA'] = df1['cbsa']
    df1['y'] = df1['ln_pre_ratio']
    df1['Constant'] = 1
    df1['FamilyxPostCovid'] = 0
    df1['Family'] = df1['v']
    df1['PostCovid'] = 0
    df1['PopxPostCovid'] = 0
    df1['PopDenxPostCovid'] = 0
    df1['IncomexPostCovid'] = 0
    df1['SFHxPostCovid'] = 0
    df1['HomeValuexPostCovid'] = 0
    df1['HousingxPostCovid'] = 0
    df1['JobsxPostCovid'] = 0
    df1 = df1[['y','Constant','FamilyxPostCovid','CBSA','Family','PostCovid',#'LogPop','LogPopDen',
                'PopxPostCovid','PopDenxPostCovid','IncomexPostCovid','SFHxPostCovid','HomeValuexPostCovid','HousingxPostCovid','JobsxPostCovid']]

    df2 = master[['cbsa','ln_post_ratio','v','ln_pop','ln_pop_den','ln_median_income',
                'ln_housing_units', 'sfh_pct', 'ln_median_housing_value','ln_jobs_per_person_20']].copy()
    df2['CBSA'] = df2['cbsa']
    df2['y'] = df2['ln_post_ratio']
    df2['Constant'] = 1
    df2['FamilyxPostCovid'] = df2['v']
    df2['Family'] = df2['v']
    df2['PostCovid'] = 1
    df2['PopxPostCovid'] = df2['ln_pop']
    df2['PopDenxPostCovid'] = df2['ln_pop_den']
    df2['IncomexPostCovid'] = df2['ln_median_income']
    df2['SFHxPostCovid'] = df2['sfh_pct']
    df2['HomeValuexPostCovid'] = df2['ln_median_housing_value']
    df2['HousingxPostCovid'] = df2['ln_housing_units']
    df2['JobsxPostCovid'] = df2['ln_jobs_per_person_20']
    df2 = df2[['y','Constant','FamilyxPostCovid','CBSA','Family','PostCovid',#'LogPop','LogPopDen',
                'PopxPostCovid','PopDenxPostCovid','IncomexPostCovid','SFHxPostCovid','HomeValuexPostCovid','HousingxPostCovid','JobsxPostCovid']]


    twfe = pd.concat([df1,df2])

    return twfe



def ipums_mig():
    usecols = ['YEAR','CBSERIAL','NUMPREC','HHWT','PERWT','HHTYPE','STATEFIP','BPL',
            'PERNUM','MOMLOC','POPLOC','MIGRATE1',
            'SEX','AGE','EMPSTAT','INCTOT','EDUCD','HHINCOME']
    df = pd.read_csv('../data/ipums_2016-2021.csv', usecols=usecols)

    df['MIG'] = df['MIGRATE1'].isin([2,3])
    # move from another state to state of birth
    df['MIG_POB'] = ((df['MIGRATE1']==3) & (df['STATEFIP'] == df['BPL']))

    temp = (df.groupby('CBSERIAL')
        .agg(NUMPREC=('NUMPREC','first'),
            MIG_COUNT = ('MIG','sum'),
            MIG_POB=('MIG_POB','sum'))
        )

    temp['IND_MOVER'] = temp[['NUMPREC','MIG_COUNT']].apply(lambda x: (x[1]>0) and ((x[1]<x[0]) or (x[0]==1 and x[1]==1)), axis=1)
    temp['FULL_HH_MOVER'] = temp[['NUMPREC','MIG_COUNT']].apply(lambda x: (x[1]==x[0]) and (x[0]>1), axis=1)
    # use this variable only for household movers
        # for individual movers, use MIG_POB
    temp['HH_MIG_POB'] = temp['MIG_POB'] > 0 

    df = df.merge(temp[['IND_MOVER','FULL_HH_MOVER','HH_MIG_POB']], left_on='CBSERIAL', right_index=True)

    cols = ['CBSERIAL','PERNUM','YEAR','HHTYPE','MOMLOC','POPLOC','MIGRATE1','FULL_HH_MOVER']
    temp = df[cols].query('HHTYPE in [1,2,3] and MIGRATE1 in [2,3] and (MOMLOC != 0 or POPLOC != 0) and not FULL_HH_MOVER')
    temp = temp.merge(df.query('HHTYPE in [1,2,3]')[['CBSERIAL','PERNUM','MIGRATE1']], 
                left_on=['CBSERIAL','MOMLOC'], right_on=['CBSERIAL','PERNUM'],
                suffixes=('','_mom'), how='left')
    temp = temp.merge(df.query('HHTYPE in [1,2,3]')[['CBSERIAL','PERNUM','MIGRATE1']], 
                left_on=['CBSERIAL','POPLOC'], right_on=['CBSERIAL','PERNUM'],
                suffixes=('','_pop'), how='left')


    def moved_to_parent(x):
        mommig = x[0]
        popmig = x[1]
        
        # no mom and pop is a non-mover
        if (mommig != mommig) & (popmig == 1):
            return True
        # both mom and pop are non-movers
        elif (mommig == 1) & (popmig == 1):
            return True
        # no pop and mom is a non-mover
        elif (mommig == 1) & (popmig != popmig):
            return True
        # either mom or pop moved as well
        else:
            return False
        
        
    temp['MOVED_TO_PARENT'] = temp[['MIGRATE1_mom','MIGRATE1_pop']].apply(lambda x: moved_to_parent(x), axis=1)
    temp = temp.query('MOVED_TO_PARENT == True')

    df = df.merge(temp[['CBSERIAL','PERNUM','MOVED_TO_PARENT']], how='left', on=['CBSERIAL','PERNUM'])

    temp = df.query('IND_MOVER')[['PERWT','MOVED_TO_PARENT','MIG_POB','YEAR']].copy()
    temp['MOVED_TO_PARENT'] = temp['PERWT'] * temp['MOVED_TO_PARENT']
    temp['MOVED_TO_POB'] = temp['PERWT'] * temp['MIG_POB'] * (~temp['MOVED_TO_PARENT'].fillna(False).astype(bool))
    temp = (temp
            .groupby('YEAR')
            .agg(
                TOTP=('PERWT','sum'), 
                MOVED_TO_PARENT=('MOVED_TO_PARENT','sum'),
                MOVED_TO_POB=('MOVED_TO_POB','sum')
            )
            .assign(
                MOVED_TO_PARENT_FRAC = lambda x: x['MOVED_TO_PARENT']/x['TOTP'] * 100,
                MOVED_TO_POB_FRAC=lambda x: x['MOVED_TO_POB']/x['TOTP'] * 100)
            )

    temp1 = df.query('FULL_HH_MOVER and HHTYPE in [1,2,3]')[['HHWT','HH_MIG_POB','YEAR']].copy()
    temp1['HH_MOVED_TO_POB'] = temp1['HHWT'] * temp1['HH_MIG_POB']
    temp1 = (temp1
            .groupby('YEAR')
            .agg(
                TOTH=('HHWT','sum'), 
                HH_MOVED_TO_POB=('HH_MOVED_TO_POB','sum')
            )
            .assign(HH_MOVED_TO_POB_FRAC=lambda x: x['HH_MOVED_TO_POB']/x['TOTH'] * 100))

    temp['HH_MOVED_TO_POB'] = temp1['HH_MOVED_TO_POB']
    temp['HH_MOVED_TO_POB_FRAC'] = temp1['HH_MOVED_TO_POB_FRAC']
    return temp




#%%

df = calc_parental_family_proxy_cbsa()
df[['v']].to_csv('../data/processed/v.csv')

phi = get_phi(df)
phi[['phi','v']].to_csv('../data/processed/phiv.csv')

#%%

cuebiq_agg = pd.read_csv('../data/cuebiq_cbsa_agg.csv', index_col=0)
df = df[['v']].merge(cuebiq_agg, right_index=True, left_index=True)
ses = make_ses_data()
master = df.drop(columns=['selected_families','total_households']).merge(ses, right_index=True, left_index=True)
master = master.reset_index(names=['cbsa'])

#%%

twfe = make_twfe_table(master)
twfe.drop(columns=['y']).to_csv('../data/processed/twfe.csv', index=False)
twfe[['y']].to_csv('../data/processed/y_twfe.csv', index=False)

#%%
ipums = ipums_mig()
ipums[['MOVED_TO_PARENT_FRAC','MOVED_TO_POB_FRAC','HH_MOVED_TO_POB_FRAC']].to_csv('../data/processed/ipums_mig.csv')

