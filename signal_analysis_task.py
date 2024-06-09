# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 20:20:22 2024

@author: Johnny McNulty

Script runs two pieces of analyses. 
1) Signal processing analysis
	- Signal cleaning: outlier removal, anomaly handling, nan handling
	- Processing: filtering, normalisation, feature extraction
	- Summary: descriptive statistics, signal visualisations
2) DNA sequence analysis
	- Simulation of errors injected into DNA sequences
	- 3 points of analysis
		- Error position distribution by error type
		- Sequence length distribution per error rate
		- GC content per error rate

Note: script checks for existence of signal datasets in the working directory,
if not found they are downloaded and extracted from my GitHub.
"""

#import libraries
import os
import requests
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, peak_widths, savgol_filter
from scipy.stats import skew, kurtosis
import random
from collections import Counter

#helper functions
def extract_zip(zip_file, extract_path):
	"""
	Extract contents of given zip file
	"""
	with zipfile.ZipFile(zip_file, 'r') as zip_ref:
		zip_ref.extractall(extract_path)

def find_and_extract_zip():
	"""
	Checks if signal_datasets.zip has already been downloaded. Contents are 
	unzipped to working directory if the file is found, otherwise file is 
	downloaded from Github and unzipped.
	"""
	
	# Check if data already downloaded
	for root, dirs, files in os.walk("."):	
		if "signal_datasets.zip" in files:			
			zip_file_path = os.path.join(root, "signal_datasets.zip")
			
			# Extract its contents if not already extracted
			if not os.path.exists("signal_datasets"):
				
				extract_zip(zip_file_path, ".")
				print("Extracted contents of 'signal_datasets.zip'")
				
			return
			
	# if folder not found, access github where data is stored
	url = 'https://github.com/JohnnyMcN/DNA_signal_analysis/raw/main/signal_datasets.zip'
	response = requests.get(url)
	
	if response.status_code == 200:
		with open('signal_datasets.zip', 'wb') as f:
			f.write(response.content)
			
		# Extract the contents of the zip folder
		extract_zip('signal_datasets.zip', '.')
		print('Downloading and extracting "signal_datasets.zip" from GitHub')
	else:
		print("Github access failed")

def insert_errors(set_n, error_rate, num_seq, seq_len, error_types, bases):
	"""
	insert_errors takes in a set of DNA sequences, and applies errors at a 
	specified rate to each sequence.
	
	Inputs: 
		set_n: list of DNA sequences, individual sequences should of dtype str
		error_rate: specified error rate, in decimal format
		num_seq: the number of sequences in a set
		seq_len: length of sequences
		error_types: list of strings corresponding to error types of insertion,
						deletion or mismatch
		bases: list of possible bases
		
	Outputs:
		set_n: modified DNA sequences
		position_tracker: locations of errors for each sequence
		type_tracker: type of errors corresponding to position tracker
		
	"""
	num_errors = int(seq_len * error_rate)
	
	type_tracker = [[] for _ in range(num_seq)]
	position_tracker = [[] for _ in range(num_seq)]
			
	#loop over each sequence in set
	for i, curr_seq in enumerate(set_n):				
		
		#convert to list for easier insert/delete/replace
		curr_seq = list(curr_seq)
		
		#loop over each error
		for j in range(num_errors):
			
			#get current errors and positions
			curr_err = random.choice(error_types)
			curr_pos = random.randint(0, len(curr_seq) - 1)
			
			type_tracker[i].append(curr_err)
			position_tracker[i].append(curr_pos)
														
			if curr_err == "insertion":
				base_to_insert = random.choice(bases)
				curr_seq.insert(curr_pos, base_to_insert)
				
			elif curr_err == "deletion":
				curr_seq.pop(curr_pos)
			
			elif curr_err == "mismatch":
				
				#remove base at current position as candidate for swap
				curr_base = curr_seq[curr_pos]
				temp_bases = [i for i in bases if i != curr_base]
				
				#replace base
				base_to_insert = random.choice(temp_bases)
				curr_seq[curr_pos] = base_to_insert
				
			else:
				print("Unrecognised error type")
				continue
		
		#return list back to string
		set_n[i] = "".join(curr_seq) 
			
	return set_n, position_tracker, type_tracker

	
#%% Signal processing analysis
def signal_processing_fn():
	"""
	Function to download and load in a dataset of signals for analysis. 
	Common signal handling steps such as data cleaning, filter, normalisation,
	and feature extraction are applied.
	
	Descriptive statistics are output to command window, with plots for signal
	visualisations and feature correlation analysis produced.
	"""
	# Check files not already in directory, otherwise extract/download
	check_files = [os.path.exists("signal_dataset_" + str(i) + ".csv") for i in range(1,8)]
	
	if all(check_files):
		print("Datasets already in directory")
	else:		
		find_and_extract_zip()
	
	#%% DATA PREPARATION: loading, merging, cleaning
	
	#initiate raw_signals with dummy row to concatenate to
	raw_signals = np.zeros((1,136))
	
	for i in range(1, 8):
		data = np.genfromtxt(''.join(["signal_dataset_", str(i) , ".csv"]), delimiter = ',')
		#if all(np.isnan(data[0,:])):
		data = data[1:,:] #remove header row
		raw_signals = np.vstack((raw_signals, data))
	
	#remove dummy row
	raw_signals = raw_signals[1:, :]
	num_signals = np.shape(raw_signals)[0]
	
	#mask signal to prevent nans affecting operations
	nan_idx = np.isnan(raw_signals)
	raw_signals = np.ma.array(raw_signals, mask = nan_idx)
	
	#detrend signals
	raw_signals = raw_signals - np.mean(raw_signals, axis = 1)[:, np.newaxis]
	
	#%% outlier and anomaly handling
	''' 
	peak finding - most signals display two prominent peaks, one positive, one negative
	consider outliers as peaks > 3 standard deviations from median peak height
	consider anomalies as signals without peaks
	'''
	
	signal_maxes = np.max(raw_signals, axis = 1)
	signal_mins = np.min(raw_signals, axis = 1)
	
	#consider peaks as those reaching 95% of signal min/max, set inter-peak distance to 10 samples to
	#ensure small fluctuations don't result in two separate peaks
	pos_peak_idx = [find_peaks(raw_signals[i,:], distance = 10, height = signal_maxes[i]*0.95)[0] for i in range(num_signals)]
	neg_peak_idx = [find_peaks(-raw_signals[i,:], distance = 10, height = -signal_mins[i]*0.95)[0] for i in range(num_signals)]
	
	#check if any signals returned 0 peaks
	zero_pos_peaks = [i for i, l in enumerate(pos_peak_idx) if len(l) == 0]
	zero_neg_peaks = [i for i, l in enumerate(neg_peak_idx) if len(l) == 0]
	
	#find all signals without at least 1 negative, or positive peak
	anomaly_set = list(set(zero_pos_peaks + zero_neg_peaks))
	
	#remove anomalies from dataset
	raw_signals = np.delete(raw_signals, anomaly_set, axis = 0)
	pos_peak_idx = [pos_peak_idx[i] for i in range(num_signals) if i not in anomaly_set]
	neg_peak_idx = [neg_peak_idx[i] for i in range(num_signals) if i not in anomaly_set]
	signal_maxes = np.delete(signal_maxes, anomaly_set)
	signal_mins = np.delete(signal_mins, anomaly_set)
	
	#update num_signals
	num_signals = np.shape(raw_signals)[0]
	
	#extract peak values
	pos_peak_vals = [raw_signals[i, pos_peak_idx[i]] for i in range(num_signals)]
	neg_peak_vals = [raw_signals[i, neg_peak_idx[i]] for i in range(num_signals)]
	
	#calculate median pos/neg peaks, apply fn. to take mean of peaks where >1 present
	pos_peak_med = np.median(list(map(lambda x: np.mean(x), pos_peak_vals)))
	neg_peak_med = np.median(list(map(lambda x: np.mean(x), neg_peak_vals)))
	
	#standard deviation of peaks
	pos_peak_sd = np.std(list(map(lambda x: np.mean(x), pos_peak_vals)))
	neg_peak_sd = np.std(list(map(lambda x: np.mean(x), neg_peak_vals)))
	
	#set threshold as mean +/- 3 standard deviations
	pos_outlier_thr = pos_peak_med + 3*pos_peak_sd
	neg_outlier_thr = neg_peak_med - 3*neg_peak_sd
	
	#find signals with peaks exceeding outlier threshold
	pos_outlier_signals = list(map(lambda i: any(raw_signals[i, :] > pos_outlier_thr), range(num_signals)))
	neg_outlier_signals = list(map(lambda i: any(raw_signals[i, :] < neg_outlier_thr), range(num_signals)))
	
	#get signal indices
	pos_outliers = np.where(np.array(pos_outlier_signals) == True)[0]
	neg_outliers = np.where(np.array(neg_outlier_signals) == True)[0]
	
	#set outlier indices to nan
	for p_o in pos_outliers:
		idx_to_nan = raw_signals[p_o, :] > pos_outlier_thr
		raw_signals[p_o, idx_to_nan] = np.nan
		
	for n_o in neg_outliers:
		idx_to_nan = raw_signals[n_o, :].data < neg_outlier_thr
		raw_signals[n_o, idx_to_nan] = np.nan
	
	#%% nan handling
	#update nan_idx array to account for outliers now set to nan
	nan_idx = np.isnan(raw_signals)
	
	#reconfigure mask and repalce nans with signal mean
	masked_signals = np.ma.array(raw_signals, mask = nan_idx)
	raw_signals = np.where(nan_idx, masked_signals.mean(axis=1)[:, np.newaxis], raw_signals)
	
	#%% SIGNAL PROCESSING: filtering, normalisation, feature extraction
	
	# savitzky-golay filter used to preserve peaks
	filtered_signals = savgol_filter(raw_signals, 5, polyorder = 3, axis = 1)
	
	# apply z-score normalisation to each signal
	signal_means = np.nanmean(filtered_signals, axis = 1)[:,np.newaxis]
	signal_stds = np.std(filtered_signals, axis = 1)[:,np.newaxis]
	norm_signals = (filtered_signals - signal_means) / signal_stds
	
	#%% feature extraction
	'''
	peak-peak difference: difference between positive peak and negative peak
		peaks of interest all seem to elicit between 40~65 samples,
		positive peak occurs prior to negative peak, negative appears more reliably prominent
		take peak-peak diff as difference between pos peak and neg peak in this region
	
	peak width (negative peaks): width of peak at half-height level
		negative peaks used as these appear more prominent
		
	power ratio: ratio of power in upper frequency half to lower half
	'''
	
	onset = 40
	offset = 65
	
	#feature arrays
	peak_diff = np.zeros((num_signals, 1))
	pwr_ratio = np.zeros((num_signals, 1))
	peak_w = np.zeros((num_signals, 1))
	
	#arrays for descriptive statistics
	peak_max = np.zeros((num_signals, 1))
	peak_min = np.zeros((num_signals, 1))
	snr = np.zeros((num_signals, 1))
	
	#signal length
	N = np.shape(norm_signals)[1]
	
	for i in range(num_signals):
		
		x = norm_signals[i,:]
		
		#find all neg peaks in current signal
		neg_peak_interval = find_peaks(-x)[0]
		
		#extract peaks within search window
		interval_filt = neg_peak_interval[(neg_peak_interval > onset) & (neg_peak_interval < offset)]
		
		#return idx and value of largest negative peak
		peak_min_loc = np.argmin(x[interval_filt])
		peak_min_loc = interval_filt[peak_min_loc]
		peak_min[i] = x[peak_min_loc]
		
		#repeat process for pos peaks, shifting search window to only look up to negative peak location
		pos_peak_interval = find_peaks(x)[0]
		interval_filt = pos_peak_interval[(pos_peak_interval > (peak_min_loc - 10)) & (pos_peak_interval < peak_min_loc)]
		
		peak_max_loc = np.argmax(x[interval_filt])
		peak_max_loc = interval_filt[peak_max_loc]	
		peak_max[i] = x[peak_max_loc]
		
		#calculate peak-peak difference
		peak_diff[i] = peak_max[i] - peak_min[i]
		
		#negative peak width
		peak_w[i] = peak_widths(-x, np.reshape(peak_min_loc, (1,) ), rel_height = 0.5)[0]
		
		#calculate signal-to-noise ratio here also, for descriptive statistics
		snr_mask = np.zeros((N))
		spike_region = np.arange(peak_max_loc - 2, peak_min_loc + 2)
		snr_mask[spike_region] = 1
		
		sig = x[snr_mask == 1]
		noise = x[snr_mask == 0]
		
		sig_power = np.sum(sig ** 2) / len(sig)
		noise_power = np.sum(noise ** 2) / len(noise)
		
		snr[i] = sig_power / noise_power
		
		#magnitude
		x_fft = np.abs(fft(x)) 
		fft_freqs = fftfreq(N)
		
		#take one sided spectrum
		x_fft = x_fft[0:N//2]
		fft_freqs = fft_freqs[0:N//2]
		
		#conver to power
		P = x_fft ** 2
		
		#set band limits
		sep_point = len(P)//4
		mid_point = len(P)//2
		
		lf_power = P[:sep_point]  #lower freq. band
		hf_power = P[sep_point:mid_point] #upper freq. band
		
		#add small term to avoid divide by zero error
		pwr_ratio[i] = np.sum(hf_power) / (np.sum(lf_power) + (10**-9))
		
		
	#compile features into array
	feat_array = np.hstack((peak_diff, peak_w, pwr_ratio))
		
	#%% Data ANALYSIS AND VISUALISATION
	
	#print descriptive statistics to command window
	signal_skewness_mean = np.mean(skew(norm_signals, axis = 1))
	signal_kurtosis_mean = np.mean(kurtosis(norm_signals, axis = 1))
	snr_db_mean = np.mean(10 * np.log(snr))
	pos_peak_mean = np.mean(peak_max)
	neg_peak_mean = np.mean(peak_min)
	
	print("\nDescriptive statistics for processed signals:")
	print("\n"
		  "No. signals removed: {}\n"
		  "No. signals with outliers: {}\n"
		  "No. signals processed: {}"
		  .format(len(anomaly_set),
			len(pos_outliers) + len(neg_outliers),
			num_signals))
	
	
	print("\nSignal mean: {:.2f}\n"
		  "Standard deviation: {:.2f}\n"
		  "Skewness: {:.2f}\n"
		  "Kurtosis: {:.2f}\n"
		  "SNR estimate (dB): {:.2f}\n"
		  "Mean peak height (pos): {:.2f}\n"
		  "Mean peak height (neg): {:.2f}"
		  .format(np.abs(np.mean(norm_signals)), 
			   np.std(norm_signals),
			   signal_skewness_mean,
			    signal_kurtosis_mean,
				 snr_db_mean,
			    pos_peak_mean,
				 neg_peak_mean))
	
	
	#to improve visualisation align signals to first signal, based on negative peak offsets
	alignment_loc = neg_peak_idx[0]
	
	offsets = [np.min(alignment_loc - neg_peak_idx[i]) for i in range(num_signals)]
	
	aligned_raw_signals = np.zeros(np.shape(raw_signals))
	
	for i in range(num_signals):
		if offsets[i] > 0:
			aligned_raw_signals[i, offsets[i]:] = raw_signals[i, :-offsets[i]]
		elif offsets[i] < 0:
			aligned_raw_signals[i, :offsets[i]] = raw_signals[i, -offsets[i]:]
		else:
			aligned_raw_signals[i,:] = raw_signals[i,:]
			
	aligned_norm_signals = np.zeros(np.shape(raw_signals))
	
	for i in range(num_signals):
		if offsets[i] > 0:
			aligned_norm_signals[i, offsets[i]:] = norm_signals[i, :-offsets[i]]
		elif offsets[i] < 0:
			aligned_norm_signals[i, :offsets[i]] = norm_signals[i, -offsets[i]:]
		else:
			aligned_norm_signals[i,:] = norm_signals[i,:]
	
	#Visualisation of raw and processed signals (unaligned and aligned)
	fig, ax = plt.subplots(4,2)
	ax = ax.flatten()
	
	raw_signals_averaged = np.mean(raw_signals, axis = 0)
	norm_signals_averaged = np.mean(norm_signals, axis = 0)
	
	aligned_raw_signals_averaged = np.mean(aligned_raw_signals, axis = 0)
	aligned_norm_signals_averaged = np.mean(aligned_norm_signals, axis = 0)
	
	ax[0].plot(np.transpose(raw_signals))
	ax[1].plot(np.transpose(norm_signals))
	
	ax[2].plot(raw_signals_averaged)
	ax[3].plot(norm_signals_averaged)
	
	ax[4].plot(np.transpose(aligned_raw_signals))
	ax[5].plot(np.transpose(aligned_norm_signals))
	
	ax[6].plot(aligned_raw_signals_averaged)
	ax[7].plot(aligned_norm_signals_averaged)
	
	ax[0].set_ylabel('Overlayed', fontweight = "bold")
	ax[2].set_ylabel('Averaged', fontweight = "bold")
	ax[4].set_ylabel('Overlayed\n(aligned)', fontweight = "bold")
	ax[6].set_ylabel('Averaged\n(aligned)', fontweight = "bold")
	
	ax[6].set_xlabel('Samples', fontweight = "bold")
	ax[7].set_xlabel('Samples', fontweight = "bold")
	
	ax[0].set_title("Raw data", fontweight = "bold")
	ax[1].set_title("Processed data", fontweight = "bold")
	fig.suptitle("Visualisation of raw and processed signals",fontweight = "bold")
	fig.tight_layout()
	
	#figure to show correlation between features
	fig, ax = plt.subplots(1, 3)
	
	cc = np.corrcoef(np.transpose(feat_array))
	
	df = pd.DataFrame(feat_array)
	df.columns = ["peak_peak","peak_width","pwr_ratio"]
	
	sns.regplot(data = df, x = "peak_peak", y = "peak_width", ax = ax[0])
	sns.regplot(data = df, x = "peak_peak", y = "pwr_ratio", ax = ax[1])
	sns.regplot(data = df, x = "peak_width", y = "pwr_ratio", ax = ax[2])
	
	ax[0].text(0.95, 0.95, "r = {:.2f} ".format(cc[0,1]), transform = ax[0].transAxes, horizontalalignment = "right", fontweight = "bold")
	ax[1].text(0.95, 0.95, "r = {:.2f} ".format(cc[0,2]), transform = ax[1].transAxes, horizontalalignment = "right", fontweight = "bold")
	ax[2].text(0.95, 0.95, "r = {:.2f} ".format(cc[1,2]), transform = ax[2].transAxes, horizontalalignment = "right", fontweight = "bold")
	
	ax[0].set_xlabel("peak-peak difference", fontweight = "bold")
	ax[1].set_xlabel("peak-peak difference", fontweight = "bold")
	ax[2].set_xlabel("peak width",fontweight = "bold")
	
	ax[0].set_ylabel("peak width", fontweight = "bold")	
	ax[1].set_ylabel("power ratio", fontweight = "bold")
	ax[2].set_ylabel("power ratio", fontweight = "bold")
	
	fig.suptitle("Analysis of feature correlations",fontweight = "bold")
	fig.tight_layout()
	
#%% DNA SEQUENCE ANALYSIS
def DNA_sequence_analysis():
	"""
	Generates random DNA sequence of 100 bases with GC content of 60%. 
	3 sets of 100 sequences are generated based off this template, with errors 
	applied to the sets at rates of 2%,5% and 10% respectively. Errors are
	applied via call to helper function insert_errors.
	
	Figures are produced showing:
		Error position distribution per error type (10% error rate)
		Sequence length distribution per error rate
		GC content per error rate
	"""
	#setup lists & constants
	GC_content = 0.6
	AT_content = 1 - GC_content
	seq_len = 100
	num_seq = 100
	GC = ["C","G"]
	AT = ["A","T"]
	bases = ["A","C","G","T"]
	error_types = ["insertion","deletion","mismatch"]
	dna_sequence = []
	
	# 60% GC content
	for i in range(int(seq_len * GC_content)):
		dna_sequence.append(random.choice(GC))
		
	# 40% AT content
	for i in range(int(seq_len * AT_content)):
		dna_sequence.append(random.choice(AT))
	
	#randomise sequence in place
	random.shuffle(dna_sequence)
	
	#join list elements to create string sequence
	template_sequence = "".join(dna_sequence)
	
	#create 3 sets, each containing 100 sequences based on template sequence
	set_1 = [template_sequence for x in range(num_seq)]
	set_2 = [template_sequence for x in range(num_seq)]
	set_3 = [template_sequence for x in range(num_seq)]
	
	#call insert errors function		
	set_1, error_positions_1, errors_1 = insert_errors(set_1, error_rate = 0.02, num_seq = num_seq, seq_len = seq_len, error_types = error_types, bases = bases)
	set_2, error_positions_2, errors_2 = insert_errors(set_2, error_rate = 0.05, num_seq = num_seq, seq_len = seq_len, error_types = error_types, bases = bases)
	set_3, error_positions_3, errors_3 = insert_errors(set_3, error_rate = 0.1, num_seq = num_seq, seq_len = seq_len, error_types = error_types, bases = bases)
		
	#%error position distribution per error type for 10% error rate (set_3)				
	error_positions_3 = np.reshape(error_positions_3, (np.size(error_positions_3), -1 ))
	errors_3 = np.reshape(errors_3, (np.size(errors_3), -1 ))
	
	idx_insertion = np.where(errors_3 == "insertion")
	idx_deletion = np.where(errors_3 == "deletion")
	idx_mismatch = np.where(errors_3 == "mismatch")
	
	insertion_array = error_positions_3[idx_insertion]
	deletion_array = error_positions_3[idx_deletion]
	mismatch_array = error_positions_3[idx_mismatch]
			
	fig, ax = plt.subplots(3,1)
	sns.histplot(insertion_array, stat = "probability", ax = ax[0], bins = 10, label = "Insertion", color = "b")
	sns.histplot(deletion_array, stat = "probability", ax = ax[1], bins = 10, label = "Deletion", color = "r")
	sns.histplot(mismatch_array, stat = "probability", ax = ax[2], bins = 10, label = "Mismatch", color = "g")
	
	#configure legend
	handles, labels = [], []
	for a in ax:
		for h, l in zip(*a.get_legend_handles_labels()):
			handles.append(h)
			labels.append(l)

	fig.legend(handles, labels, loc = "upper center", bbox_to_anchor = (0.5, 0.95), ncol = 3)
	
	ax[0].set_xticks([])
	ax[1].set_xticks([])
	ax[2].set_xlabel("Error position", fontweight = "bold")
	[ax[i].set_yticks([0.05, 0.1, 0.15]) for i in range(3)]
	[ax[i].set_ylabel("Probability", fontweight = "bold") for i in range(3)]
				   
	fig.suptitle("Distribution of error position by type (10% error rate)", fontweight = "bold")
	
	#sequence length distribution per error rate
	l1 = [len(set_1[i]) for i in range(num_seq)]
	l2 = [len(set_2[i]) for i in range(num_seq)]
	l3 = [len(set_3[i]) for i in range(num_seq)]
	
	fig, ax = plt.subplots(1,1)
	
	sns.boxplot(data = [l1, l2, l3], ax = ax)
		
	ax.set_xticklabels(["2%","5%","10%"], fontweight = "bold")
	ax.set_xlabel("Error rate", fontweight = "bold")
	ax.set_ylabel("Sequence length", fontweight = "bold")
	ax.set_title("Distribution of sequence length by error rate", fontweight = "bold")
	
	#GC content per error rate
	GC_1 = np.zeros(100)
	GC_2 = np.zeros(100)
	GC_3 = np.zeros(100)
	
	for i in range(100):
		d_1 = Counter(set_1[i])
		GC_1[i] = (d_1["G"] + d_1["C"]) / len(set_1[i])
		
		d_2 = Counter(set_2[i])
		GC_2[i] = (d_2["G"] + d_2["C"]) / len(set_2[i])	
		
		d_3 = Counter(set_3[i])
		GC_3[i] = (d_3["G"] + d_3["C"]) / len(set_3[i])	
	
		
	fig, ax = plt.subplots(1,1)
		
	sns.kdeplot(GC_1*100, ax = ax, label = "2%")
	sns.kdeplot(GC_2*100, ax = ax, label = "5%")
	sns.kdeplot(GC_3*100, ax = ax, label = "10%")
	
	ax.legend()	
	ax.set_xlabel("GC content (%)", fontweight = "bold")
	ax.set_ylabel("Density", fontweight = "bold")
	ax.set_title("GC content per error rate", fontweight = "bold")
	
	
#%% main function
def main():
	"""
	main function used simply to call signal_processing_fn, and 
	DNA_sequence_analysis
	"""
	signal_processing_fn()
	DNA_sequence_analysis()
	
	
#call main function
main()
plt.show()