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

AGE_BUCKET = AGE + "_bucket"
AGE_BUCKET_BOUNDARIES = [0, 20, 40, 100] # refer pandas.cut() for syntax on binning
AGE_LABELS = ["young", "middle-age", "old"] # refer pandas.cut() for syntax on labels

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


def extract_metric_from_base_metrics(metrics_df:pd.DataFrame, metric:str="FPR", rating_class:int=1) -> pd.Series:
    return metrics_df.apply(lambda m: m["fairness_metrics_per_class"][metric][rating_class], axis=1)


def extract_ad_category_id(s:str):
    """Converts ids of the form A08_09 to 8"""
    raw_category_id = s.split("_")[0] 
    return int(raw_category_id[1:])

def test_extract_ad_category_id():
    assert extract_ad_category_id("A08_09") == 8
    assert extract_ad_category_id("A12_19") == 12
    logging.info("All tests passed")
    
test_extract_ad_category_id()


ad_categories = pd.read_csv("data/AdCats.csv")


def fetch_gender_metrics(inference_df) -> Dict[str, pd.Series]:
    gfm = GroupFairnessMetrics(inference_df, GENDER)

    fpr = extract_metric_from_base_metrics(gfm.fetch_base_metrics(), "FPR")
    eod = gfm.equal_opportunity_difference("F", "M")
    aod = gfm.average_odds_difference("F", "M")
    
    metrics = {FPR: fpr, EOD: pd.Series({"Priv=F, UnPriv=M": eod}), AOD: pd.Series({"Priv=F, UnPriv=M": aod})}
    return metrics


def fetch_age_metrics(inference_df:pd.DataFrame) -> Dict[str, pd.Series]:
    inference_df[AGE] = inference_df[AGE].astype("int")
    inference_df[AGE_BUCKET] = pd.cut(inference_df[AGE], bins=AGE_BUCKET_BOUNDARIES, labels=AGE_LABELS)
    gfm = GroupFairnessMetrics(inference_df, AGE_BUCKET)

    fpr = extract_metric_from_base_metrics(gfm.fetch_base_metrics())

    eod_ym = gfm.equal_opportunity_difference("young", "middle-age")
    eod_yo = gfm.equal_opportunity_difference("young", "old")

    aod_ym = gfm.average_odds_difference("young", "middle-age")
    aod_yo = gfm.average_odds_difference("young", "old")

    metrics = {
        FPR: fpr, 
        EOD: pd.Series(
            {"Priv=young, UnPriv=midle-age": eod_ym, "Priv=young, UnPriv=old": eod_yo}
        ),
        AOD: pd.Series(
            {"Priv=young, UnPriv=middle-age": aod_ym, "Priv=young, UnPriv=old": aod_yo}
        )
    }
    return metrics


def fetch_model_metrics_for_feature_on_dataset(model_id:str, feature:str, ad_category:str):
    df = pd.read_csv(f"data/model-logs/{model_id}/inference_data.csv")
    df["AdCategoryId"] = df.apply(lambda row: extract_ad_category_id(row["AdId"]), axis=1)
    df = pd.merge(df, ad_categories, left_on="AdCategoryId", right_on="AdCatId", suffixes=("", "_y"), validate="many_to_one")
    total_count = df.shape[0]
    
    if ad_category.lower() != "all":
        df = df[df[ADCATNAME] == ad_category]
        
    feature_metrics_map = {
        GENDER: fetch_gender_metrics,
        AGE: fetch_age_metrics
    }
    
    metrics = feature_metrics_map[feature](df)
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


common_opts = opts(show_grid=True, height=200, width=300) #, tools=["hover"]
gfm_opts = opts(ylabel="Closer to zero, the better", ylim=(-1, 1))
fpr_opts = opts(ylim=(0, 1))


def chart_for_metric(state:Dict, metric:str, viz_opts=opts()):
    bar_charts = []
    for g in state[M1][metric].keys():
        chart_data = [(state[model]["label"], state[model][metric][g]) for model in state.keys()]
        bar_chart = hv.Bars(chart_data, hv.Dimension(f'Models for {g}'), " ").opts(common_opts).opts(viz_opts)
        bar_charts.append(bar_chart)

    return hv.Layout(bar_charts).opts(shared_axes=False, title=metric).cols(1) # TODO: Title disturbs the layout


class AdsFairnessExplorer(param.Parameterized):
    protected_features = [GENDER, AGE]
    ad_types = ['All', 'Kitchen & Home', 'Office Products', 'Dating Sites',
       'Clothing & Shoes', 'Betting', 'Musical Instruments', 'Grocery',
       'DIY & Tools', 'Toys & Games', 'Media (BMVD)',
       'Jewellery & Watches', 'Health & Beauty', 'Sports & Outdoors',
       'Pet Supplies', 'Console & Video Games', 'Consumer Electronics',
       'Automotive', 'Computer Software', 'Garden & Outdoor living',
       'Baby Products']
    
    protected_feature = param.Selector(sorted(protected_features), default=GENDER)
    ad_category = param.Selector(sorted(ad_types), default='All')

    @param.depends('protected_feature', "ad_category")
    def make_view(self):
        state = fetch_all_models_metrics_for_feature_on_dataset(self.protected_feature, self.ad_category)
        total_records_count = reduce(lambda acc, model_stats: model_stats["total_count"] + acc, state.values(), 0)
        plotted_records_count = reduce(lambda acc, model_stats: model_stats["plot_count"] + acc, state.values(), 0)
        num_models = len(state.values())
        html_stats_div = lambda s: f"""<p style="text-align: center; margin-top: 3rem;">{s}</p>"""
        stats_text = html_stats_div(f"{self.ad_category} ad category has {plotted_records_count / num_models:.2f} records of {total_records_count / num_models:.0f} records in total on an average")
        
        gspec = pn.GridSpec(sizing_mode="stretch_both", align="center") # background="gray"
        gspec[:15, :2] = chart_for_metric(state, FPR, fpr_opts)
        gspec[:15, 2:4] = chart_for_metric(state, EOD, gfm_opts)
        gspec[:15, 4:6] = chart_for_metric(state, AOD, gfm_opts)
        gspec[15, :] =  stats_text
        
        return gspec

explorer = AdsFairnessExplorer()


with open("header.html.partial", "r") as f:
    header = f.read()


dashboard = pn.Column(pn.pane.HTML(header), explorer.param, explorer.make_view)
dashboard.servable(title="Fairness in Ads Dashboard")




