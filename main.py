import torch
import pandas as pd
import datasets_read

df = datasets_read.NCMAPSS(dataset_name='DS03-012', dir_path='/home/linyuhan/Dokumente/Masterarbeit/dataset/Turbofan_Engine_Degradation_Simulation_Dataset2',
                            downsampling=10).load_data()
print(df['unit'].tail())

RUL = datasets_read.NCMAPSS(dataset_name='DS03-012', dir_path='/home/linyuhan/Dokumente/Masterarbeit/dataset/Turbofan_Engine_Degradation_Simulation_Dataset2',
                            downsampling=10)._getitem_RUL(index=1)
print(RUL)
