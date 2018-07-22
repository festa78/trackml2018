#!/usr/bin/python3

import os

import glob
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
from metric_learn import *
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import DBSCAN, MeanShift
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.decomposition import PCA
import torch
import tqdm

from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

import operator


def cart2spherical(cart):
    r = np.linalg.norm(cart, axis=0)
    theta = np.degrees(np.arccos(cart[2] / r))
    phi = np.degrees(np.arctan2(cart[1], cart[0]))
    #theta = np.arccos(cart[2] / r)
    #phi = np.arctan2(cart[1], cart[0])
    return np.vstack((r, theta, phi))


def parse(truth, min_num=5, max_num=16):
    truth_dedup = truth.drop_duplicates('particle_id')
    #truth_sort = truth_dedup.sort_values('weight', ascending=False)

    p_traj_list = []
    p_traj_clean_list = []
    for i, (_, tr) in enumerate(truth_dedup.iterrows()):
        p_traj = truth[truth.particle_id == tr.particle_id][[
            'tx', 'ty', 'tz'
        ]].reset_index(drop=True)
        p_id = pd.DataFrame(
            np.ones((p_traj.shape[0], 1), dtype=np.int) * i, columns=['id'])
        p_traj_id = pd.concat((p_traj, p_id), axis=1)
        p_traj_list.append(p_traj_id)

        if min_num < p_traj.shape[0] and p_traj.shape[0] < max_num:
            p_traj_clean_list.append(p_traj_id)

    p_traj_df = pd.concat(p_traj_list, ignore_index=True)
    p_traj_clean_df = pd.concat(p_traj_clean_list, ignore_index=True)

    # Convert to spherical coordinate.
    rtp_list = []
    rtp_clean_list = []
    for i, p_traj in enumerate(p_traj_list):
        xyz = p_traj.loc[:, ['tx', 'ty', 'tz']].values.transpose()

        rtp = cart2spherical(xyz).transpose()
        rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))
        rtp_list.append(rtp_df)

        if min_num < p_traj.shape[0] and p_traj.shape[0] < max_num:
            rtp_clean_list.append(rtp_df)

    rtp_df = pd.concat(rtp_list, ignore_index=True)
    rtp_clean_df = pd.concat(rtp_clean_list, ignore_index=True)

    p_traj_df = pd.concat((p_traj_df, rtp_df), axis=1, ignore_index=False)
    p_traj_clean_df = pd.concat(
        (p_traj_clean_df, rtp_clean_df), axis=1, ignore_index=False)
    return p_traj_df, p_traj_clean_df


def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id] * len(hits), hits.hit_id.values,
                                labels))
    submission = pd.DataFrame(
        data=sub_data, columns=["event_id", "hit_id", "track_id"]).astype(int)
    return submission


def compute_x2(xyz, prefix='t'):
    x = xyz[prefix + 'x'].values
    y = xyz[prefix + 'y'].values
    z = xyz[prefix + 'z'].values

    r = np.sqrt(x**2 + y**2 + z**2)
    xyz[prefix + 'x2'] = x / r
    xyz[prefix + 'y2'] = y / r
    r = np.sqrt(x**2 + y**2)
    xyz[prefix + 'z2'] = z / r


p_traj_clean_all = []
hits_all = []
end_id = 0
for i, (event_id, hits, cells, particles, truth) in tqdm.tqdm(
        enumerate(load_dataset('../input/train_1', skip=0))):
    _, p_traj_clean_df = parse(truth)
    p_traj_clean_df['id'] += end_id
    end_id = p_traj_clean_df['id'].values[-1]
    compute_x2(p_traj_clean_df)
    compute_x2(hits, prefix='')
    p_traj_clean_all.append(p_traj_clean_df)

    xyz = hits.loc[:, ['x', 'y', 'z']].values.transpose()
    rtp = cart2spherical(xyz).transpose()
    rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))
    hits = pd.concat((hits, rtp_df), axis=1)

    hits_all.append(hits)
    if i > -1:
        break

scl = preprocessing.StandardScaler()

X_cols = ('r', 'theta', 'phi', 'x', 'y', 'z', 'x2', 'y2', 'z2')
tX_cols = ('r', 'theta', 'phi', 'tx', 'ty', 'tz', 'tx2', 'ty2', 'tz2')
# X_cols = ('x2', 'y2', 'z2')
# tX_cols = ('tx2', 'ty2', 'tz2')

p_traj_clean_cat = pd.concat(p_traj_clean_all, ignore_index=True)
hits_cat = pd.concat(hits_all, ignore_index=True)

X_scale = scl.fit_transform(hits_cat.loc[:, X_cols].values)
X_clean_scale = scl.transform(p_traj_clean_cat.loc[:, tX_cols].values)
Y_clean_scale = p_traj_clean_cat['id'].values

for i, idx in enumerate(np.unique(Y_clean_scale)):
    Y_clean_scale[Y_clean_scale == idx] = i

traj_max = 5000
X_clean_scale_trun = X_clean_scale[Y_clean_scale < traj_max]
Y_clean_scale_trun = Y_clean_scale[Y_clean_scale < traj_max]

num_classes = np.unique(Y_clean_scale_trun).size

# Metric learning via learning classification task.
# X_scale_th = torch.tensor(
#     X_scale, dtype=torch.float, device=torch.device('cuda:0'))
X_clean_scale_th = torch.tensor(
    X_clean_scale_trun, dtype=torch.float, device=torch.device('cuda:0'))
Y_clean_scale_th = torch.tensor(
    Y_clean_scale_trun, dtype=torch.long, device=torch.device('cuda:0'))

# Simple model definition.
model = torch.nn.Sequential(
    torch.nn.Linear(len(tX_cols), 20), torch.nn.ReLU(), torch.nn.Linear(20, 20),
    torch.nn.ReLU(), torch.nn.Linear(20, 20), torch.nn.ReLU(), torch.nn.Linear(20, 20),
    torch.nn.ReLU(), torch.nn.Linear(20, num_classes)).to(
        torch.device('cuda:0'))

loss_fn = torch.nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=.1)
schedular = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5000, 10000, 50000])
for i in range(100000):
    optimizer.zero_grad()
    output = model(X_clean_scale_th)
    loss = loss_fn(output, Y_clean_scale_th)
    loss.backward()
    optimizer.step()
    schedular.step()
    if i % 100 == 0:
        print(i, loss.item())
        torch.save(model.state_dict(), './model_9_20180731.pth')

# torch.save(model.state_dict(), './model_9_20180728.pth')
model.load_state_dict(torch.load('./model_9_20180728.pth'))

pca = PCA(n_components=100)
output = model(X_clean_scale_th)
pca.fit(output.cpu().detach().numpy())

eps_best = 0
score_best = 0

for event_id, hits, cells, particles, truth in load_dataset(
        '../input/train_1', skip=0):
    xyz = hits.loc[:, ['x', 'y', 'z']].values.transpose()
    rtp = cart2spherical(xyz).transpose()
    rtp_df = pd.DataFrame(rtp, columns=('r', 'theta', 'phi'))
    hits = pd.concat((hits, rtp_df), axis=1)
    compute_x2(hits, prefix='')

    X_scale = scl.transform(hits.loc[:, X_cols].values)
    X_scale_th = torch.tensor(
        X_scale, dtype=torch.float, device=torch.device('cuda:0'))
    X_trans = model(X_scale_th).cpu().detach().numpy()
    X_trans = pca.transform(X_trans)
    for eps in [
            # 0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.003, 0.005, 0.007, 0.01,
            # 0.03, 0.05, 0.07, 0.1, 0.3, 0.5, 0.7, 0.9, 1., 1.3, 1.5, 1.7, 2., 5,
            # 7, 9, 11, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
            65, 70, 75, 80, 85, 90, 95, 100,
            110, 120, 130 ,140, 150, 160, 170, 180, 190, 200,
            210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
            350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
            850, 900, 950, 1000
    ]:
        db = DBSCAN(eps=eps, min_samples=1, metric='euclidean', n_jobs=-1)
        # db = MeanShift()
        out = db.fit(X_trans)
        # out = db.fit(X_scale)

        labels = out.labels_
        one_submission = create_one_event_submission(event_id, hits, labels)
        score = score_event(truth, one_submission)
        print('eps {}, Event {}, score {}'.format(eps, event_id, score))
        if score > score_best:
            score_best = score
            eps_best = eps
    break

df_test = []
id_total = 1
eps = eps_best

test_list = np.unique(
    [p.split('-')[0] for p in sorted(glob.glob('../input/test/**'))])
for test in test_list:
    hits = load_event(test, parts=['hits'])[0]
    hits['event_id'] = int(test[-9:])

    compute_x2(hits, prefix='')

    X_scale = scl.transform(hits.loc[:, X_cols].values)
    X_scale_th = torch.tensor(
        X_scale, dtype=torch.float, device=torch.device('cuda:0'))
    X_trans = model(X_scale_th).cpu().detach().numpy()
    X_trans = pca.transform(X_trans)

    db = DBSCAN(eps=eps, min_samples=1, algorithm='auto', n_jobs=-1)
    hits['particle_id'] = db.fit_predict(X_trans) + id_total
    id_total += len(hits['particle_id'].unique())
    df_test.append(hits[['event_id', 'hit_id', 'particle_id']].copy())
    print(test, len(hits['particle_id'].unique()))

df_test_all = pd.concat(df_test, ignore_index=True)
sub = pd.read_csv('../input/sample_submission.csv')
sub = pd.merge(sub, df_test_all, how='left', on=['event_id', 'hit_id'])
#print(sub['particle_id'])
sub['track_id'] = sub['particle_id'].astype('int64')
sub[['event_id', 'hit_id', 'track_id']].to_csv(
    '20180710-submission-test.csv', index=False)
print(sub[['event_id', 'hit_id', 'track_id']].head())
