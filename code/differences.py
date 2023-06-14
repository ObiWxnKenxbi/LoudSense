#VoS = Voice on Set
#SS = Source Separation
#D = Demucs

import pandas as pd

#Differences between VoS and SS

csv_file_name = "/../../results/msr_results.csv"

df = pd.read_csv(csv_file_name, dtype={'VoS-M-L': float, 'VoS-S-L': float, 'VoS-SMR': float, 'SS-M-L': float, 'SS-S-L': float, 'SS-SMR': float, 'D-S-L': float, 'D-M-L': float, 'D-SMR': float})

df['VoS-S vs SS-S'] = df['VoS-S-L'] - df['SS-S-L']
df['VoS-M vs SS-M'] = df['VoS-M-L'] - df['SS-M-L']
df['VoS-SMR vs SS-SMR'] = df['VoS-SMR'] - df['SS-SMR']

df = df.drop(columns=["Language","SS-M-L","SS-S-L", "SS-SMR", "VoS-M-L", "VoS-S-L", "VoS-SMR", "D-S-L", "D-M-L", "D-SMR"])

df.head()

#Differences between VoS and D
df = pd.read_csv(csv_file_name, dtype={'VoS-M-L': float, 'VoS-S-L': float, 'VoS-SMR': float, 'SS-M-L': float, 'SS-S-L': float, 'SS-SMR': float, 'D-S-L': float, 'D-M-L': float, 'D-SMR': float})

df['VoS-S vs D-S'] = df['VoS-S-L'] - df['D-S-L']
df['VoS-M vs D-M'] = df['VoS-M-L'] - df['D-M-L']
df['VoS-SMR vs D-SMR'] = df['VoS-SMR'] - df['D-SMR']

df = df.drop(columns=["Language","SS-M-L","SS-S-L", "SS-SMR", "VoS-M-L", "VoS-S-L", "VoS-SMR", "D-S-L", "D-M-L", "D-SMR"])

df.head()

#Differences between SS and D
df = pd.read_csv(csv_file_name, dtype={'VoS-M-L': float, 'VoS-S-L': float, 'VoS-SMR': float, 'SS-M-L': float, 'SS-S-L': float, 'SS-SMR': float, 'D-S-L': float, 'D-M-L': float, 'D-SMR': float})

df['SS-S vs D-S'] = df['SS-S-L'] - df['D-S-L']
df['SS-M vs D-M'] = df['SS-M-L'] - df['D-M-L']
df['SS-SMR vs D-SMR'] = df['SS-SMR'] - df['D-SMR']

df = df.drop(columns=["Language","SS-M-L","SS-S-L", "SS-SMR", "VoS-M-L", "VoS-S-L", "VoS-SMR", "D-S-L", "D-M-L", "D-SMR"])

df.head()
