# IDC409_Project_SpeechToText
This project provides a Python program for real-time speech recognition using a microphone or audio files (SpeechToText_Updated.py). The program includes two main functionalities:
1. Speech-to-Text using Microphone Input: Converts real-time spoken words to text. 
2. Audio-to-Text from File: Processes pre-recorded audio files to extract spoken text, which is then displayed and optionally saved to a text file.
   

This repository contains a Jupyter notebook, titled S2T_Model.ipynb, written as past of the assessment project for IDC 409.


Discussion:
The dataset consists of over 13100 (.wav) audio clips  which can be used to train the model. Due to time and computation restrictions, we were not able to train the model much. Training even one epoch, was approximated to be atleast 20+hours on Jupyter Notebook. Therefore, the dataset was restricted to only ~130 files for an epoch, which led to an untrained model, and inaccurate predictions. With more training, on larger datasets, it should be able to predict the text with a word error rate of less than 30%. 

data_url = https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

Google link of trained model - https://drive.google.com/drive/folders/1BuushBPBlmXWSanC2egCbpkZD9WSMbQt?usp=drive_link
