# Panel校验
# 基于ROC进行验证
import pandas as pd,numpy as np, scipy as sp ,sklearn as sk
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import classification_report


df_pc = pd.read_csv('cross_pc_panel.csv', '\x01', header = -1)
df_mobile = pd.read_csv('cross_mobile_panel.csv','\x01', header = -1)

df_pc.columns = ['device_predict', 'cookie_panel', 'rank']
df_mobile.columns = ['device_panel', 'cookie_predict', 'rank']

panel_pc = pd.read_csv('panel_pc.csv', header = -1)
panel_mobile = pd.read_csv('panel_mobile.csv',header = -1)

panel_pc.columns = ['admckid', 'vendor_id' ,'gender' ,'age']
panel_mobile.columns = ['idfa', 'imei', 'androidid','opeudid', 'mac', 'cookie_mobile', 'vendor_id', 'gender' ,'age']

# 按照筛选规则，筛选，生成一列可以被拿来做join的列 mobileid

cross_device_device = df_pc.set_index('device_predict').join(df_mobile.set_index('device_panel'), rsuffix = '_device_cookie_rank', how = 'inner')
cross_device_device = cross_device_device.drop_duplicates()
cross_device_device = cross_device_device.reset_index()
cross_device_device.columns = ['device_panel', u'cookie_panel', u'rank', u'cookie_predict',u'rank_device_cookie_rank']


cross_cookie_cookie = df_mobile.set_index('cookie_predict').join(df_pc.set_index('cookie_panel'), rsuffix = '_cookie_device_rank', how = 'inner')
cross_cookie_cookie = cross_cookie_cookie.drop_duplicates()
# cookie , device 知道性别年龄， device_check_mobile_panel 是通过模型推出来的跨设备ID
cross_cookie_cookie = cross_cookie_cookie.reset_index()
cross_cookie_cookie.columns = ['cookie_panel', u'device_panel', u'rank', u'device_predict',u'rank_cookie_device_rank']

cross_device_device['cross'] = cross_device_device['cookie_panel'].astype('str') + cross_device_device['device_panel']
cross_cookie_cookie['cross'] = cross_cookie_cookie['cookie_panel'].astype('str') + cross_cookie_cookie['device_panel']
temp = cross_cookie_cookie.set_index('cross').join(cross_device_device.set_index('cross'), how = 'inner', rsuffix = '_other')


cross_cookie_cookie[['gender_pc','age_pc']] = cross_cookie_cookie.set_index('cookie_panel').join(panel_pc.set_index('admckid')[['gender','age']])

cross_cookie_cookie[['gender_mobile','age_mobile']] = cross_cookie_cookie.set_index('device_panel').join(panel_mobile.set_index('mobileid')[['gender','age']])












