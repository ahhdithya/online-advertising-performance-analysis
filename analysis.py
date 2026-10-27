from pathlib import Path
import argparse
import base64
import html
import json
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
METRICS = ['displays', 'cost', 'clicks', 'revenue', 'post_click_conversions', 'post_click_sales_amount']
DAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def divide(a, b):
    """Undefined rates remain NaN, not zero or infinity."""
    return a / b.replace(0, np.nan)


def summarize(df, keys):
    g = df.groupby(keys, dropna=False, observed=True)[METRICS].sum()
    g['rows'] = df.groupby(keys, dropna=False, observed=True).size()
    g['active_days'] = df.groupby(keys, dropna=False, observed=True)['date'].nunique()
    g['ctr_pct'] = 100 * divide(g.clicks, g.displays)
    g['conversion_rate_pct'] = 100 * divide(g.post_click_conversions, g.clicks)
    g['firm_cpc_usd'] = divide(g.cost, g.clicks)
    g['company_x_cpc_usd'] = divide(g.revenue, g.clicks)
    g['firm_cpa_usd'] = divide(g.cost, g.post_click_conversions)
    g['company_x_cpa_usd'] = divide(g.revenue, g.post_click_conversions)
    g['firm_roi_pct'] = 100 * divide(g.revenue - g.cost, g.cost)
    g['company_x_roas'] = divide(g.post_click_sales_amount, g.revenue)
    g['displays_per_day'] = divide(g.displays, g.active_days)
    g['clicks_per_day'] = divide(g.clicks, g.active_days)
    g['conversion_share_pct'] = 100 * g.post_click_conversions / df.post_click_conversions.sum()
    return g


def load_data(path):
    raw = pd.read_csv(path)
    blank = raw.columns[raw.isna().all()].tolist()
    df = raw.drop(columns=blank).copy()
    required = ['month', 'day', 'campaign_number', 'user_engagement', 'banner', 'placement'] + METRICS
    if not set(required).issubset(df.columns):
        raise ValueError('Missing required columns: ' + str(set(required) - set(df.columns)))
    duplicates = int(df.duplicated().sum())
    for c in ['month', 'campaign_number', 'user_engagement', 'banner', 'placement']:
        df[c] = df[c].astype('string').str.strip()
    df['placement'] = df.placement.fillna('Unknown').replace('', 'Unknown')
    for c in METRICS + ['day']:
        df[c] = pd.to_numeric(df[c], errors='raise')
    if df[METRICS].isna().any().any() or (df[METRICS] < 0).any().any():
        raise ValueError('Missing or negative metric values require review.')
    month = df.month.map({'April': 4, 'May': 5, 'June': 6})
   
    df['date'] = pd.to_datetime(dict(year=2020, month=month, day=df.day), errors='raise')
    df['weekday'] = df.date.dt.day_name()
    df['day_type'] = np.where(df.date.dt.dayofweek >= 5, 'Weekend', 'Weekday')
    dims = df.banner.str.extract(r'^(\d+)\s*x\s*(\d+)$').astype(int)
    df['banner_area'] = dims[0] * dims[1]
    df['time_index'] = (df.date - pd.Timestamp('2020-04-01')).dt.days
    df['weekday_number'] = df.date.dt.dayofweek
    audit = {'source_rows': len(raw), 'empty_columns_removed': blank,
             'missing_placement_rows': int(raw.placement.isna().sum()),
             'exact_duplicate_rows_retained': duplicates,
             'conversions_above_clicks_rows': int((df.post_click_conversions > df.clicks).sum()),
             'conversions_without_clicks_rows': int(((df.clicks == 0) & (df.post_click_conversions > 0)).sum()),
             'clicks_above_displays_rows': int((df.clicks > df.displays).sum()),
             'observed_days': int(df.date.nunique()),
             'date_min': str(df.date.min().date()), 'date_max': str(df.date.max().date())}
    return df, audit


def train_models(df, out):
    """Predict row-level clicks conditional on a supplied impression volume.
    April training -> May model choice -> April+May refit -> June test.
    No cost, revenue, conversions, or sales features are used.
    """
    cats = ['campaign_number', 'user_engagement', 'banner', 'placement']
    nums = ['log_displays', 'banner_area', 'weekday_number', 'time_index']
    x = df[cats + nums[1:]].copy()
    x['log_displays'] = np.log1p(df.displays)
    april = df.date < '2020-05-01'
    may = (df.date >= '2020-05-01') & (df.date < '2020-06-01')
    june = df.date >= '2020-06-01'
    rows = []
    def score(stage, name, mask, pred):
        y = df.loc[mask, 'clicks']
        rows.append(dict(stage=stage, model=name, rows=int(mask.sum()),
                         MAE=mean_absolute_error(y, pred), RMSE=np.sqrt(mean_squared_error(y, pred)),
                         R2=r2_score(y, pred), actual_clicks=int(y.sum()), predicted_clicks=float(np.sum(pred))))
    def baseline(train, test):
        rate = df.loc[train, 'clicks'].sum() / df.loc[train, 'displays'].sum()
        return df.loc[test, 'displays'].to_numpy() * rate
    def make_model(leaves):
        prep = ColumnTransformer([('categories', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats),
                                  ('numeric', 'passthrough', nums)])
        return make_pipeline(prep, HistGradientBoostingRegressor(loss='poisson', max_leaf_nodes=leaves,
                             max_iter=150, learning_rate=0.06, l2_regularization=1.0,
                             early_stopping=False, random_state=42))
    score('May validation', 'Training CTR baseline', may, baseline(april, may))
    models = {}
    for leaves in [7, 15]:
        name = f'Poisson boosting ({leaves} leaves)'
        models[name] = leaves
        m = make_model(leaves).fit(x.loc[april], df.loc[april, 'clicks'])
        pred = m.predict(x.loc[may])
        pred[df.loc[may, 'displays'].to_numpy() == 0] = 0
        score('May validation', name, may, pred)
    best = min(rows, key=lambda r: r['MAE'])['model']
    train = april | may
    pred = baseline(train, june)
    score('June test', 'Training CTR baseline', june, pred)
    if best in models:
        m = make_model(models[best]).fit(x.loc[train], df.loc[train, 'clicks'])
        pred = m.predict(x.loc[june])
        pred[df.loc[june, 'displays'].to_numpy() == 0] = 0
        score('June test', best, june, pred)
    result = pd.DataFrame(rows)
    result.to_csv(out / 'model_metrics.csv', index=False)
    predictions = df.loc[june, ['date', 'campaign_number', 'user_engagement', 'banner', 'placement', 'displays', 'clicks']].copy()
    predictions['predicted_clicks'] = pred
    predictions.to_csv(out / 'june_predictions.csv', index=False)
    return result, best, predictions


def run(csv_path=ROOT / 'data/online_advertising_performance_data.csv', output=ROOT / 'outputs'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    df, audit = load_data(csv_path)
    df.to_csv(out / 'cleaned_data.csv', index=False)
    (out / 'data_quality.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    groups = {'daily': ['date'], 'monthly': ['month'], 'campaign': ['campaign_number'],
              'banner': ['banner'], 'placement': ['placement'], 'engagement': ['user_engagement'],
              'weekday': ['weekday'], 'day_type': ['day_type'],
              'banner_engagement': ['banner', 'user_engagement'],
              'campaign_banner_placement': ['campaign_number', 'banner', 'placement'],
              'campaign_banner': ['campaign_number', 'banner'],
              'campaign_engagement': ['campaign_number', 'user_engagement'],
              'monthly_campaign': ['month', 'campaign_number'], 'monthly_banner': ['month', 'banner']}
    tables = {name: summarize(df, keys) for name, keys in groups.items()}
    tables['monthly'] = tables['monthly'].reindex(['April', 'May', 'June'])
    tables['weekday'] = tables['weekday'].reindex(DAYS)
    for name, t in tables.items():
        t.to_csv(out / f'{name}.csv')
    daily, monthly = tables['daily'], tables['monthly']
    camp, banner, placement = [tables[k] for k in ['campaign', 'banner', 'placement']]
    total = df[METRICS].sum()
    total.to_csv(out / 'totals.csv', header=['total'])
   
    flags = pd.DataFrame(index=df.index)
    thresholds = {}
    for c in ['cost', 'clicks', 'revenue']:
        q1, q3 = df[c].quantile([.25, .75])
        cutoff = q3 + 1.5 * (q3 - q1)
        flags[c + '_high_iqr'] = df[c] > cutoff
        thresholds[c] = float(cutoff)
    flagged = pd.concat([df, flags], axis=1).loc[flags.any(axis=1)]
    flagged.to_csv(out / 'outliers.csv', index_label='source_zero_based_row')
    (out / 'outlier_thresholds.json').write_text(json.dumps(thresholds, indent=2))
    sensitivity = pd.concat({'all_rows': df[METRICS].sum(),
                             'exact_duplicates_removed': df.drop_duplicates()[METRICS].sum()}, axis=1)
    sensitivity['difference'] = sensitivity.all_rows - sensitivity.exact_duplicates_removed
    sensitivity.to_csv(out / 'duplicate_sensitivity.csv')
    corr = pd.DataFrame({'row_pearson': [df.cost.corr(df.revenue)],
                         'row_spearman': [df.cost.corr(df.revenue, method='spearman')],
                         'daily_pearson': [daily.cost.corr(daily.revenue)]}, index=['cost_vs_revenue'])
    corr.to_csv(out / 'correlations.csv')
    eng_mix = df.pivot_table(index='month', columns='user_engagement', values='displays', aggfunc='sum').reindex(['April', 'May', 'June'])
    eng_mix = 100 * eng_mix.div(eng_mix.sum(axis=1), axis=0)
    eng_mix.to_csv(out / 'monthly_engagement_impression_share.csv')
    banner_mix = df.pivot_table(index='banner', columns='user_engagement', values='displays', aggfunc='sum').fillna(0)
    banner_mix = 100 * banner_mix.div(banner_mix.sum(axis=1), axis=0)
    banner_mix.to_csv(out / 'banner_engagement_impression_share.csv')
    day_mix = df.pivot_table(index='day_type', columns='user_engagement', values='displays', aggfunc='sum')
    day_mix = 100 * day_mix.div(day_mix.sum(axis=1), axis=0)
    day_mix.to_csv(out / 'weekday_weekend_engagement_mix.csv')
    for name in ['campaign', 'banner']:
        t = tables['monthly_' + name].reset_index()
        t['roas_month_rank'] = t.groupby('month').company_x_roas.rank(ascending=False, method='min')
        t['firm_roi_month_rank'] = t.groupby('month').firm_roi_pct.rank(ascending=False, method='min')
        t.to_csv(out / f'{name}_monthly_consistency.csv', index=False)
        tables[name + '_consistency'] = t
    metrics, best, predictions = train_models(df, out)
    plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlesize': 12, 'font.size': 10})
    charts = []
    def save(fig, name):
        fig.tight_layout()
        path = out / (name + '.png')
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        charts.append(path)
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, col, label in zip(axes, ['displays', 'clicks', 'post_click_sales_amount'], ['Displays', 'Clicks', 'Attributed sales (USD)']):
        ax.plot(daily.index, daily[col], color='#bed9e5', label='Daily')
        ax.plot(daily.index, daily[col].rolling(7, min_periods=7).mean(), color='#126782', label='7-day mean')
        ax.set_ylabel(label); ax.legend(loc='upper right')
    axes[0].set_title('Campaign volume and attributed sales • April–June 2020')
    save(fig, '01_daily_trends')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    banner.sort_values('clicks').clicks.plot.barh(ax=axes[0], color='#126782', title='Clicks by banner')
    banner.sort_values('ctr_pct').ctr_pct.plot.barh(ax=axes[1], color='#209d82', title='CTR by banner (%)')
    save(fig, '02_banner_performance')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    placement[['displays', 'clicks']].plot.bar(ax=axes[0], logy=True, title='Placement volume (log scale)')
    placement.conversion_rate_pct.plot.bar(ax=axes[1], color='#126782', title='Transactions per 100 clicks')
    save(fig, '03_placement_performance')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    eng_mix.plot.bar(stacked=True, ax=axes[0], title='Audience mix: share of displays (%)')
    monthly.ctr_pct.plot.bar(ax=axes[1], color='#126782', title='Overall CTR by month (%)')
    save(fig, '04_engagement')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].scatter(df.cost, df.revenue, s=6, alpha=.2, color='#126782')
    axes[0].set(xlabel='Firm placement cost (USD)', ylabel='Firm click revenue (USD)', title='Cost and revenue: row-level relationship')
    tables['weekday'].conversion_rate_pct.plot.bar(ax=axes[1], color='#209d82', title='Transactions per 100 clicks by weekday')
    save(fig, '05_cost_and_weekday')
    fig, ax = plt.subplots(figsize=(11, 4))
    predictions.groupby('date')[['clicks', 'predicted_clicks']].sum().plot(ax=ax, title='June holdout: actual and predicted clicks')
    ax.set_ylabel('Clicks'); save(fig, '06_model_predictions')
 
    parts = ['<!doctype html><html><head><meta charset="utf-8"><title>Online Advertising Analysis</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 22px;color:#18313e}h1,h2{line-height:1.2}h1{font-size:38px}h2{margin-top:42px;color:#126782}table{border-collapse:collapse;font-size:13px;width:100%;margin:15px 0}td,th{border-bottom:1px solid #d9e3e8;padding:7px;text-align:right}th{background:#edf4f6}.scroll{overflow-x:auto}img{max-width:100%}.note{background:#edf4f6;padding:18px;border-left:4px solid #126782}nav a{display:block}code{background:#edf4f6}footer{margin-top:50px}</style></head><body><h1>Online Advertising Performance</h1><p>Company X / Advert Firm A • 1 April–30 June 2020 • USD</p>']
    def p(text): parts.append('<p>' + html.escape(text) + '</p>')
    def table(t, columns=None, limit=None):
        if columns is not None: t = t[columns]
        if limit is not None: t = t.head(limit)
        parts.append('<div class="scroll">' + t.to_html(float_format=lambda v: f'{v:,.4f}', border=0) + '</div>')
    def section(n, title, text, t=None, cols=None):
        parts.append(f'<h2 id="q{n}">{n}. {html.escape(title)}</h2>'); p(text)
        if t is not None: table(t, cols)
    def top(t, col): return str(t[col].idxmax())
    def low(t, col): return str(t[col].idxmin())
    parts.append('<div class="note">')
    p('Revenue belongs to Advert Firm A; it is Company X’s recorded ad spend. Firm CPC = cost/clicks; Company X CPC = revenue/clicks. Firm ROI = (revenue−cost)/cost. Company X ROAS = post-click sales/revenue. Company X profit ROI cannot be calculated without product margins and other costs. All rates use sums of numerators divided by sums of denominators.')
    p('Engagement is a pre-existing audience category, not an observed continuous score. CTR is used as a behavioral proxy. Conversion rate here means attributed transactions per click, not the probability that a unique person converts. The file has transactions greater than clicks in some rows and no unique-user IDs. Attribution is over 30 days: the extraction date and maturity of June outcomes are unknown. Results describe associations, not causal effects.')
    parts.append('</div>')
    p(f"Data checks: {len(df):,} rows, {audit['observed_days']} dates, 3 campaigns, 8 banners; {audit['missing_placement_rows']} missing placements retained as Unknown. Two empty columns removed. {audit['exact_duplicate_rows_retained']} exact duplicate rows retained because no unique record ID establishes whether they are errors; duplicate_sensitivity.csv shows their impact. {audit['conversions_above_clicks_rows']} rows have transactions above clicks and {audit['conversions_without_clicks_rows']} have transactions without clicks; {audit['clicks_above_displays_rows']} have clicks above displays. Flag these for source-owner review.")
    table(total.to_frame('Overall total'))
    section(1, 'Overall engagement trend', f"CTR fell from {monthly.loc['April','ctr_pct']:.2f}% in April to {monthly.loc['June','ctr_pct']:.2f}% in June. Audience mix and activity volume also changed; this does not prove individuals became less engaged.", monthly, ['displays','clicks','ctr_pct'])
    table(eng_mix)
    section(2, 'Banner size and clicks', f"{top(banner,'clicks')} generated the most clicks; {top(banner,'ctr_pct')} had the highest CTR. Raw clicks depend on impression volume, audience and placement, so banner area alone cannot establish an effect.", banner.sort_values('clicks',ascending=False), ['displays','clicks','ctr_pct'])
    section(3, 'Placements with most displays and clicks', f"{top(placement,'displays')} led displays and {top(placement,'clicks')} led clicks. These are anonymous codes; website versus app cannot be inferred.", placement.sort_values('clicks',ascending=False), ['displays','clicks','ctr_pct'])
    section(4, 'Cost–revenue correlation', 'Pearson measures linear association; Spearman measures rank association. Shared traffic volume can drive both cost and revenue. A correlation does not show that increasing spend causes higher returns.', corr)
    section(5, 'Average revenue per click', f"Total firm revenue / total clicks = ${total.revenue:,.4f} / {total.clicks:,.0f} = ${total.revenue/total.clicks:.4f} per click. For Company X this is average ad spend per click, not revenue earned. Attributed sales per click = ${total.post_click_sales_amount/total.clicks:.4f}.")
    section(6, 'Campaigns with highest conversion rates', f"{top(camp,'conversion_rate_pct')} led the aggregate transaction-per-click rate. Compare click volume and attribution maturity as well as the rate.", camp.sort_values('conversion_rate_pct',ascending=False), ['clicks','post_click_conversions','conversion_rate_pct'])
    section(7, 'Post-click sales patterns', f"Monthly attributed sales were ${monthly.loc['April','post_click_sales_amount']:,.2f}, ${monthly.loc['May','post_click_sales_amount']:,.2f}, and ${monthly.loc['June','post_click_sales_amount']:,.2f}. The highest daily sales were on {daily.post_click_sales_amount.idxmax().date()}. These are sales attributed to campaign dates, not necessarily transaction dates.", monthly, ['post_click_sales_amount','post_click_conversions','company_x_roas'])
    section(8, 'Audience engagement across banners', 'The table gives the percentage of each banner’s displays served to each engagement category. This measures audience allocation, not a change in user behavior. banner_engagement.csv also provides CTR and conversion rates within each category.', banner_mix)
    section(9, 'Placement conversion rates', f"{top(placement,'conversion_rate_pct')} had the highest overall transaction-per-click rate. Unknown is retained in totals but is not a known publisher type.", placement.sort_values('conversion_rate_pct',ascending=False), ['clicks','post_click_conversions','conversion_rate_pct'])
    section(10, 'Seasonal patterns in displays and clicks', 'Compare daily averages to avoid the unequal number of weekdays. The series shows only one quarter: it supports descriptive weekday patterns and a time trend, not annual or reliably recurring seasonal claims. Changes in campaign mix can explain apparent weekday differences.', tables['weekday'], ['active_days','displays_per_day','clicks_per_day','ctr_pct'])
    section(11, 'Engagement category and revenue', 'Do not encode Low/Medium/High as equally spaced numbers and interpret a Pearson correlation as definitive. Compare category totals and exposure-normalized rates instead. Revenue differences may reflect volume, targeting or click pricing.', tables['engagement'], ['displays','clicks','revenue','ctr_pct','company_x_cpc_usd'])
    section(12, 'Outliers requiring review', f"A global upper IQR rule flags {len(flagged):,} rows for cost, clicks or revenue. This skewed dataset contains large-volume segments, so these are review flags, not grounds for deletion. See outliers.csv for all flagged rows and flags; the table shows the 10 largest costs.", flagged.sort_values('cost',ascending=False).head(10), ['date','campaign_number','banner','placement','displays','cost','clicks','revenue'])
    eligible = tables['campaign_banner_placement'].query('clicks >= 1000 and post_click_conversions >= 30')
    section(13, 'Campaign effectiveness by banner and placement', 'Below are the top 10 combinations by Company X ROAS, restricted to at least 1,000 clicks and 30 transactions. These are analyst-selected support thresholds, not statistical significance tests. Very high ROAS requires tracking and attribution validation before budget decisions. The full combination table is in campaign_banner_placement.csv.', eligible.sort_values('company_x_roas',ascending=False).head(10), ['clicks','post_click_conversions','ctr_pct','conversion_rate_pct','company_x_cpa_usd','company_x_roas'])
    section(14, 'ROI and consistency', f"{top(camp,'firm_roi_pct')} led firm ROI; {top(camp,'company_x_roas')} led Company X ROAS. {top(banner,'firm_roi_pct')} led banner firm ROI and {top(banner,'company_x_roas')} led banner ROAS. Overall leaders are not automatically consistent winners: monthly rank tables below show every campaign and banner. Lower rank is better; 1 is the monthly leader.", camp, ['cost','revenue','firm_roi_pct','company_x_roas'])
    for name in ['campaign', 'banner']:
        table(tables[name+'_consistency'], ['month', 'campaign_number' if name=='campaign' else 'banner', 'clicks','firm_roi_pct','firm_roi_month_rank','company_x_roas','roas_month_rank'])
    section(15, 'Conversion distribution by placement', 'Transaction shares sum to 100%, including Unknown. Shares describe where recorded transactions occurred, while conversion rates adjust for click volume.', placement.sort_values('post_click_conversions',ascending=False), ['post_click_conversions','conversion_share_pct','conversion_rate_pct'])
    section(16, 'Weekday versus weekend engagement', 'Compare the impression-weighted audience mix and CTR. Per-day activity is shown because there are more weekdays than weekend days. These are descriptive differences, without a claim of statistical significance.', tables['day_type'], ['active_days','displays_per_day','clicks_per_day','ctr_pct','conversion_rate_pct'])
    table(day_mix)
    section(17, 'CPC across campaigns and banner sizes', 'Two perspectives are shown: firm placement cost per click and Company X ad spend per click. Use ratios of totals, not an average of row CPCs. campaign_banner.csv includes the joint breakdown.', camp, ['clicks','firm_cpc_usd','company_x_cpc_usd'])
    table(banner, ['clicks','firm_cpc_usd','company_x_cpc_usd'])
    section(18, 'Cost-effective campaigns and placements', f"Using Company X CPA, {low(camp,'company_x_cpa_usd')} is the lowest-cost campaign. Unknown has the smallest raw placement CPA but only 5 clicks and 2 transactions, so it is not a reliable winner. Among named placements with at least 1,000 clicks and 30 transactions, {low(placement.drop(index='Unknown', errors='ignore').query('clicks >= 1000 and post_click_conversions >= 30'),'company_x_cpa_usd')} has the lowest CPA. Profitability also depends on margins and customer value.", camp, ['post_click_conversions','firm_cpa_usd','company_x_cpa_usd'])
    table(placement, ['post_click_conversions','firm_cpa_usd','company_x_cpa_usd'])
    section(19, 'Conversion rates by day of week', f"{top(tables['weekday'],'conversion_rate_pct')} had the highest observed transaction-per-click rate. This could reflect campaign mix or time trends; it is not evidence that shifting all spend to that day improves conversions.", tables['weekday'], ['clicks','post_click_conversions','conversion_rate_pct'])
    section(20, 'Campaign effectiveness by engagement type', 'Within-campaign comparisons separate some campaign-mix effects. Compare transaction rates, CPA and ROAS, but audience selection and placement differences still prevent causal interpretation.', tables['campaign_engagement'], ['clicks','post_click_conversions','conversion_rate_pct','company_x_cpa_usd','company_x_roas'])
    parts.append('<h2>Machine learning: conditional click prediction</h2>')
    p('Task: predict clicks for a campaign/banner/placement/audience/date, given an impression volume. Actual displays are available in this offline evaluation; for a future scenario you must supply a planned display count. This is not an unconditional forecast or a user-level conversion classifier. Features: audience and campaign categories, banner, placement, log(1+displays), banner area, weekday, and days since 1 April. Target: clicks. Excluded: cost, revenue, conversions, and sales, because they would leak outcomes or be unavailable beforehand.')
    p(f'April trains the candidates, May chooses the smallest MAE, and June is held out. The candidates are an exposure-based baseline (training CTR × displays) and two Poisson gradient-boosting models. Selected on May: {best}. The selected method is refit on April+May before June evaluation. Lower MAE and RMSE are better; R² may be negative under distribution shift. June is not used for model choice. No claim of production readiness is made; later-period data and rolling validation are needed.')
    table(metrics)
    test_result = metrics.iloc[-1]
    p(f"June calibration check: the selected method predicts {test_result.predicted_clicks:,.0f} clicks versus {test_result.actual_clicks:,.0f} actual, a {(test_result.predicted_clicks / test_result.actual_clicks - 1) * 100:.1f}% difference. Better row-level error does not guarantee accurate total forecasts.")
    parts.append('<h2>Charts</h2>')
    for path in charts:
        parts.append('<img alt="' + html.escape(path.stem) + '" src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() + '">')
    parts.append('<footer>Source: supplied online_advertising_performance_data.csv. Generated by analysis.py. Undefined rates appear as NaN. Full numerical tables and source audit accompany this report.</footer></body></html>')
    (out / 'report.html').write_text('\n'.join(parts), encoding='utf-8')
    print(json.dumps(audit, indent=2)); print('\nMONTHLY RESULTS\n',monthly[['displays','clicks','ctr_pct','post_click_sales_amount']].to_string())
    print('\nMODEL RESULTS\n', metrics.to_string(index=False))
    print('\nOpen:', out / 'report.html')
    return df, tables, metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/online_advertising_performance_data.csv')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    run(args.data, args.output)
