#!/usr/bin/env python
# coding: utf-8

import warnings, logging
from typing import Dict
from pprint import pprint
from collections import defaultdict
from functools import reduce

import param
import panel as pn

from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

import holoviews as hv

import bokeh
from bokeh.plotting import show
from bokeh.io import output_notebook

from holoviews.operation.datashader import datashade, shade, dynspread, rasterize
from holoviews.streams import RangeXY
from holoviews.operation import decimate
from holoviews import opts, Cycle

hv.extension('bokeh')
renderer = hv.renderer('bokeh')
output_notebook()

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(levelname)s %(filename)s] %(message)s')


ADID = "AdId"
AGE = "Age"
GENDER = "Gender"
INCOME = "Income"
HOMECOUNTRY = "Homecountry"
RATING = "Rating"
PREDICTED_RATING = "pred_rating"
PRED_CONFIDENCE = "pred_confidence"
ADCATEGORYID = "AdCategoryId"
ADCATID = "AdCatId"
ADCATNAME = "AdCatName"
ADCATNUMADS = "AdCatNumAds"

EOD = "Equal Opportunity Difference"
AOD = "Average Odds Difference"
FPR = "False Positive Rate"

M1 = "1589009535"
M2 = "1589144135" 
M3 = "1589150656"


# Credits: https://stackoverflow.com/a/50671617/1585523 has good explanations for all metrics and is from where the code has been copied

def metrics_from_df(df:pd.DataFrame, confidence_threshold=0):
    """Drop examples with probability < confidence_threshold from calc"""
    y_true = df[RATING]
    y_pred = df[PREDICTED_RATING]
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    return {
        "TPR": TPR, "TNR": TNR, "PPV": PPV, "NPV": NPV, "FPR": FPR, "FNR": FNR, "FDR": FDR, "ACC": ACC
    }


class GroupFairnessMetrics:
    def __init__(self, model_inference_data:pd.DataFrame, protected_feature:str):
        """
        Compute fairness metrics between 2 groups of population - privileged & unprivileged
        Based on your dataset you could use,
            "Female" to privileged_grp_label and 
            "Male" to be unprivileged_grp_label
        for Gender as the protected_feature
        All metrics are computed on model_inference_data which has RATING and PREDICTED_RATING values for each row
        """
        self._df, self._pf, = model_inference_data, protected_feature
        self._base_metrics = "fairness_metrics_per_class"
        self._pf_metrics_df = self._df.groupby(self._pf).apply(metrics_from_df).to_frame(self._base_metrics)
    
    def fetch_base_metrics(self):
        return self._pf_metrics_df
    
    def equal_opportunity_difference(self, pg_lbl:str, upg_lbl:str, rating_class=1):
        r"""TPR{unprivileged} - TPR{privileged} ideally should be zero"""
        upg_opp = self._pf_metrics_df.loc[upg_lbl][self._base_metrics]["TPR"][rating_class]
        pg_opp = self._pf_metrics_df.loc[pg_lbl][self._base_metrics]["TPR"][rating_class]
        return upg_opp - pg_opp

    def statistical_parity_difference(self):
        raise NotImplementedError("TODO")

    def average_odds_difference(self, pg_lbl:str, upg_lbl:str, rating_class=1):
        """Average of difference in FPR and TPR for unprivileged and privileged groups"""
        tpr_diff = self.equal_opportunity_difference(pg_lbl, upg_lbl, rating_class)
        
        upg_fpr = self._pf_metrics_df.loc[upg_lbl][self._base_metrics]["FPR"][rating_class]
        pg_fpr = self._pf_metrics_df.loc[pg_lbl][self._base_metrics]["FPR"][rating_class]
        fpr_diff = upg_fpr - pg_fpr
        
        return 0.5 * (fpr_diff + tpr_diff)

    def disparate_impact():
        raise NotImplementedError("TODO")

    def theil_index():
        raise NotImplementedError("TODO")


def plot_for_metric_class(metric_df:pd.DataFrame, metric:str="FPR", rating_class:int=1):
    """Generates plot for metric and given rating_class from metric_df indexed by dimension of interest"""
    plot_df = metric_df.apply(lambda m: m["fairness_metrics_per_class"][metric][rating_class], axis=1)
    plot_df = plot_df.reset_index().rename({0: metric}, axis=1)
    return plot_df


def extract_ad_category_id(s:str):
    """Converts ids of the form A08_09 to 8"""
    raw_category_id = s.split("_")[0] 
    return int(raw_category_id[1:])

def test_extract_ad_category_id():
    assert extract_ad_category_id("A08_09") == 8
    assert extract_ad_category_id("A12_19") == 12
    print("All tests passed")
    
test_extract_ad_category_id()


ad_categories = pd.read_csv("data/AdCats.csv")


def fetch_gender_metrics(inference_df):
    gfm = GroupFairnessMetrics(inference_df, GENDER)

    fpr = plot_for_metric_class(gfm.fetch_base_metrics())
    eod = gfm.equal_opportunity_difference("F", "M")
    aod = gfm.average_odds_difference("F", "M")
    
    metrics = {FPR: fpr, EOD: eod, AOD: aod}
    return metrics


def fetch_model_metrics_for_feature_on_dataset(model_id:str, feature:str, ad_category:str):
    df = pd.read_csv(f"data/model-logs/{model_id}/inference_data.csv")
    df["AdCategoryId"] = df.apply(lambda row: extract_ad_category_id(row["AdId"]), axis=1)
    df = pd.merge(df, ad_categories, left_on="AdCategoryId", right_on="AdCatId", suffixes=("", "_y"), validate="many_to_one")
    total_count = df.shape[0]
    
    if ad_category.lower() != "all":
        df = df[df[ADCATNAME] == ad_category]
    
    metrics = fetch_gender_metrics(df)
    metrics.update({"plot_count": df.shape[0], "total_count": total_count})
    return metrics


def fetch_all_models_metrics_for_feature_on_dataset(feature:str, ad_category:str):
    state = {
        M1: { "label": "Base" },
        M2: { "label": "M2" },
        M3: { "label": "M3" }
    }
    for model_id in state.keys():
        state[model_id].update(fetch_model_metrics_for_feature_on_dataset(
            model_id, feature, ad_category
        ))
    return state


common_opts = opts(show_grid=True) #, tools=["hover"]
gfm_opts = opts(ylabel="Closer to zero, the better", height=300, ylim=(-1, 1))
fpr_opts = opts(ylim=(0, 1), height=200)


class AdsFairnessExplorer(param.Parameterized):
    protected_features = [GENDER]
    ad_types = ['All', 'Kitchen & Home', 'Office Products', 'Dating Sites',
       'Clothing & Shoes', 'Betting', 'Musical Instruments', 'Grocery',
       'DIY & Tools', 'Toys & Games', 'Media (BMVD)',
       'Jewellery & Watches', 'Health & Beauty', 'Sports & Outdoors',
       'Pet Supplies', 'Console & Video Games', 'Consumer Electronics',
       'Automotive', 'Computer Software', 'Garden & Outdoor living',
       'Baby Products']
    
    protected_feature = param.Selector(sorted(protected_features), default=GENDER)
    ad_category = param.Selector(sorted(ad_types), default='All')

    def make_fpr(self, state:Dict):
        fpr_bars = []
        for g in ["F", "M"]:
            fpr_data = []
            for model in state.keys():
                name, fpr = state[model]["label"], state[model][FPR]
                fpr_data.append((name, fpr[fpr[GENDER] == g]["FPR"].tolist()[0]))
            fpr_bar = hv.Bars(fpr_data, hv.Dimension(f'For {g}'), FPR).opts(common_opts).opts(fpr_opts)
            fpr_bars.append(fpr_bar)

        fpr_chart = hv.Layout(fpr_bars).opts(opts.Layout(shared_axes=False, title=" ")).cols(1) # TODO: Title disturbs the layout
        return fpr_chart

    def make_eod(self, state:Dict):
        eod_data = [(state[m]["label"], state[m][EOD]) for m in state.keys()]
        eod_bars = hv.Bars(eod_data, hv.Dimension('Priv=F, Unpriv=M'), " ").opts(common_opts).opts(gfm_opts).opts(title=EOD)
        return eod_bars
    
    def make_aod(self, state:Dict):
        aod_data = [(state[m]["label"], state[m][AOD]) for m in state.keys()]
        aod_bars = hv.Bars(aod_data, hv.Dimension('Priv=F, Unpriv=M'), " ").opts(common_opts).opts(gfm_opts).opts(title=AOD)
        return aod_bars
    
    @param.depends('protected_feature', "ad_category")
    def make_view(self):
        state = fetch_all_models_metrics_for_feature_on_dataset(self.protected_feature, self.ad_category)
        total_records_count = reduce(lambda acc, model_stats: model_stats["total_count"] + acc, state.values(), 0)
        plotted_records_count = reduce(lambda acc, model_stats: model_stats["plot_count"] + acc, state.values(), 0)
        num_models = len(state.values())
        html_stats_div = lambda s: f"""<p style="text-align: center; margin-top: 3rem;">{s}</p>"""
        
        gspec = pn.GridSpec(width=900, sizing_mode='scale_height')
        gspec[0, 0] = self.make_fpr(state)
        gspec[0, 1] = self.make_eod(state)
        gspec[0, 2] = self.make_aod(state)
        gspec[1, :] = html_stats_div(f"{self.ad_category} ad category has {plotted_records_count / num_models:.2f} records of {total_records_count / num_models:.0f} records in total on an average")

        return gspec

explorer = AdsFairnessExplorer()


gspec = pn.GridSpec(sizing_mode="scale_both")
gspec[0, :2] = pn.Row("## Fairness in Ads Dashboard")
gspec[1, :1] = explorer.param
gspec[2, :2] = explorer.make_view

gspec.servable(title="Fairness in Ads Dashboard")




