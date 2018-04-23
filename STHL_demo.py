# -*- coding: utf-8 -*-
"""
Created on Wed Apr 05 14:40:59 2017

@author: lwang
"""

import mne
import numpy as np
import seaborn as sns
import pandas as pd

import os
import fnmatch
import pylab as pl
from scipy import io
from scipy.stats.stats import pearsonr 

from STHL_demo_funs import plot_single
from STHL_demo_funs import plot_pairwithSpeech

data_path = 'D:/EEG_Data/Ex012 (STHL Algorithm)/'
result_path = 'C:/Users/lwang/My Work/EEG_Experiments/Results/Ex012 (STHL Algorithm)/'

# EEG data analysis example
subj = 'C003'
     
bdfs = fnmatch.filter(os.listdir(data_path), '*' + 'STHL_' + '*MBclicks' + '*bpnoise*' +  subj + '.bdf')
fname = bdfs[0][:-4]

raw = mne.io.read_raw_edf(data_path + fname + '.bdf', preload=True)

raw.plot(n_channels=32, scalings=100e-6)

allvars = io.loadmat(result_path + 'Le_20151215_STHL_MBclicks_300ms_nonoise_70dB_C003_cpcaENV_ALL.mat')

mspec = allvars['mspec_SN']
freq = allvars['f']
length = np.round(allvars['Fs']*0.3)
mspec_dB = 20*np.log10(mspec)/2 + 123.5 - 20*np.log10(np.sqrt(length))

fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(freq.squeeze(), mspec_dB[0,:])
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('EEG strength (dB)')
ax.set_xlim((0, 1000))

# EEG measure - x1
allvars = io.loadmat(result_path + 'STHL_MBclicks_300ms_bpnoise_70dB_SpecENV_allsubj.mat')
subs_FFR = allvars['subjlist']
mspec_SN_allsub = allvars['mspec_SN_allsub']
length = np.round(allvars['Fs']*0.3)
FFR_strdB = 20*np.log10(mspec_SN_allsub.sum(axis=1))/2 + 123.5 - 20*np.log10(np.sqrt(length))
x1 = FFR_strdB[0,...]
plot_single(x=x1, x_label='EEG', x_subs=subs_FFR)
print pearsonr(x1[0,:], x1[1,:])
print pearsonr(x1[0,:], x1[2,:])
print pearsonr(x1[1,:], x1[2,:])

# ITD measure - x2
allvars = io.loadmat(result_path + 'itd_jnd_3bands_allsub.mat')
subjlist = allvars['subjlist']
thresh_3bands = allvars['thresh_3bands']
subs_itd = np.delete(subjlist, -3)
tmp = [x[0] for x in subs_itd]
subs_itd = np.asarray(tmp)
x2 = np.delete(thresh_3bands, -3, axis=1)
plot_single(x=x2, x_label='ITD', x_subs=subs_itd)
print pearsonr(x2[0,:], x2[1,:])
print pearsonr(x2[0,:], x2[2,:])
print pearsonr(x2[1,:], x2[2,:])

# Audiogram measure - x3
allvars = io.loadmat(result_path + 'audiogram.mat')
subs_audiogram = allvars['subjlist']
tmp = [x[0] for x in subs_audiogram]
subs_audiogram = np.asarray(tmp).squeeze()
subs_audiogram = np.delete(subs_audiogram, 30)
audio_th = allvars['audio_valid'].T
audio_th = (audio_th[::2,:]+audio_th[1::2,:])/2
audio_th = np.delete(audio_th, 30, axis=1)
audiogram_freqs = [125, 250, 500, 1000, 2000, 4000, 8000]
fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(audiogram_freqs, audio_th)
ax.set_xlabel('Audiogram frequency')
ax.set_ylabel('Hearing Threshold')
ax.set_ylim((-20,60))
ax.invert_yaxis()
ax.set_xscale('log')
ax.set_xticks(np.array(audiogram_freqs))
ax.set_xticklabels(np.array(audiogram_freqs))
x3 = np.zeros((3, len(subs_audiogram)))
x3[0,:] = audio_th[:4,:].mean(axis=0)
x3[1,:] = audio_th[4,:]
x3[2,:] = audio_th[5:,:].mean(axis=0)
plot_single(x=x3, x_label='Hearing Threshold', x_subs=subs_audiogram)
print pearsonr(x3[0,:], x3[1,:])
print pearsonr(x3[0,:], x3[2,:])
print pearsonr(x3[1,:], x3[2,:])


# Speech measure - y
allvars = io.loadmat(result_path + 'LiSN_summary.mat')
subjlist = allvars['subjlist']
subs_lisn = np.delete(subjlist, 0)
tmp = [x[0] for x in subs_lisn]
subs_lisn = np.asarray(tmp)
lisn_meansrt = allvars['LiSN_meansrt'].T
lisn_meansrt = np.delete(lisn_meansrt, 0, axis=1)
y = lisn_meansrt[0,:]


# Gender for subjects
allvars = io.loadmat(result_path + 'gender.mat')
subs_gender = allvars['subjlist']
tmp = [x[0][0] for x in subs_gender]
subs_gender = np.asarray(tmp)
gender = allvars['gender']
tmp = [x[0][0][0][0] for x in gender]
gender = np.asarray(tmp)

# Speech vs. Gender
comm_subs = np.intersect1d(subs_lisn, subs_gender)
gender_comm = gender[np.in1d(subs_gender, comm_subs)]
speech_comm = y[np.in1d(subs_lisn, comm_subs)]
pl.figure()
sns.barplot(x = gender_comm, y = speech_comm)
sns.plt.title('Speech Performance vs. Gender')

#### pairwise correlation
# EEG vs. Speech
plot_pairwithSpeech(x1=x1, x1_label='EEG', x1_subs=subs_FFR, x2=y, x2_label='Speech Threshold', x2_subs=subs_lisn)


# ITD vs. Speech
plot_pairwithSpeech(x1=x2, x1_label='ITD', x1_subs=subs_itd, x2=y, x2_label='Speech Threshold', x2_subs=subs_lisn)


# Hearing_Level vs. Speech
plot_pairwithSpeech(x1=x3, x1_label='Hearing Threshold', x1_subs=subs_audiogram, x2=y, x2_label='Speech Threshold', x2_subs=subs_lisn)


# save all data into a dataframe
subs = [subs_FFR, subs_itd, subs_audiogram, subs_lisn, subs_gender]
result = set(subs[0])
for s in subs[1:]:
    result.intersection_update(s)
print result

comm_subs = list(result)
nsub = len(comm_subs)
x1_df = x1[:,np.in1d(subs_FFR, comm_subs)]
x2_df = x2[:,np.in1d(subs_itd, comm_subs)]
x3_df = x3[:,np.in1d(subs_audiogram, comm_subs)]
y_df = y[np.in1d(subs_lisn, comm_subs)]
gender_df = gender[np.in1d(subs_gender, comm_subs)]

subs_df = comm_subs

data =  pd.DataFrame({ 'Subject' : subs_df,
                       'EEG1' : x1_df[0,:],
                       'EEG2' : x1_df[1,:],
                       'EEG3' : x1_df[2,:],
                       'ITD1' : x2_df[0,:],
                       'ITD2' : x2_df[1,:],
                       'ITD3' : x2_df[2,:],
                       'HL1' : x3_df[0,:],
                       'HL2' : x3_df[1,:],
                       'HL3' : x3_df[2,:],
                       'Speech' : y_df,                       
                       'Gender' : gender_df
                     })

fresult = 'C:/Users/lwang/My Work/EEG_Experiments/Results/Ex012 (STHL Algorithm)/'
data.to_csv(fresult+'STHL_dataframe.csv')

#### use a classifier 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

df=data.copy()
del df['Subject']
del df['Gender']

df.loc[data['Speech']>-2, 'Speech'] = 1
df.loc[data['Speech']<=-2, 'Speech'] = 0

        
label = df.pop('Speech')

# full model with all 9 predictors        
nrep = 200
nstep = 20
train_score = np.zeros((nrep,nstep))
test_score = np.zeros((nrep,nstep))
for test_size in range(1,nstep+1):
    for rep in range(1,nrep):
        
        data_train, data_test, label_train, label_test = train_test_split(df, label, test_size = test_size)    
        
        regr = LogisticRegression(C=0.1,penalty='l1')        
        regr.fit(data_train, label_train)
        
#        svm = SVC(kernel='rbf')
#        svm.fit(data_train, label_train)
#        regr = svm
        
        train_score[rep-1,test_size-1] = regr.score(data_train, label_train)
        test_score[rep-1,test_size-1] = regr.score(data_test, label_test)                

fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(range(30-1,30-nstep-1,-1),train_score.mean(axis=0),'b')
ax.plot(range(30-1,30-nstep-1,-1),test_score.mean(axis=0),'r')
ax.legend(['train score', 'test score'])
ax.set_xlabel('Training size')
ax.set_ylabel('Classification Accuracy')
ax.set_ylim((0.5, 1))


# simplified model with 3 predictors 
df_simp=data[['EEG3', 'ITD1', 'ITD3', 'Speech']]
df_simp.loc[data['Speech']>-2, 'Speech'] = 1
df_simp.loc[data['Speech']<=-2, 'Speech'] = 0   

label = df_simp.pop('Speech').astype('int')

nrep = 200
nstep = 20
train_score = np.zeros((nrep,nstep))
test_score = np.zeros((nrep,nstep))
for test_size in range(1,nstep+1):
    for rep in range(1,nrep):
        
        data_train, data_test, label_train, label_test = train_test_split(df_simp, label, test_size = test_size)    
        
        regr = LogisticRegression(penalty='l1')
        regr.fit(data_train, label_train)
        
#        svm = SVC(kernel='linear', C=1)
#        svm.fit(data_train, label_train)
#        regr = svm
        
        train_score[rep-1,test_size-1] = regr.score(data_train, label_train)
        test_score[rep-1,test_size-1] = regr.score(data_test, label_test)        

fig = pl.figure()
ax = fig.add_subplot(111)
ax.plot(range(30-1,30-nstep-1,-1),train_score.mean(axis=0),'b')
ax.plot(range(30-1,30-nstep-1,-1),test_score.mean(axis=0),'r')
ax.legend(['train score', 'test score'])
ax.set_xlabel('Training size')
ax.set_ylabel('Classification Accuracy')
ax.set_ylim((0.5, 1))
