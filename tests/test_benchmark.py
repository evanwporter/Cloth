import numpy as np
import pandas as pd
import cloth
import Sloth as sl 

data = np.random.rand(100, 10)
columns = ['col_0', 'col_1', 'col_2', 'col_3', 'col_4', 'col_5', 'col_6', 'col_7', 'col_8', 'col_9']
rows = list(str(i) for i in range(100))

pdf = pd.DataFrame(data, rows, columns)
df = cloth.DataFrame(data, rows, columns)
sdf = sl.DataFrame(data, rows, columns)

def test_benchmark_pandas_iloc(benchmark):
    benchmark(lambda: pdf.iloc[10:40])

def test_benchmark_cloth_iloc(benchmark):
    benchmark(lambda: df.iloc[10:40])

def test_benchmark_sloth_iloc(benchmark):
    benchmark(lambda: sdf.iloc[10:40])

def test_benchmark_pandas_gt(benchmark):
    benchmark(lambda: df[df.col_0 > .5])

def test_benchmark_cloth_gt(benchmark):
    benchmark(lambda: df[df.col_0 > .5])