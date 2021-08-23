#from pandas.plotting import scatter_matrix
import pandas.plotting
import matplotlib.pyplot as plt
import pandas

plt.close("all")

#ax = plt.gca()
#ax2 = plt.twinx()
#ax.set_ylabel('runtime (s)')
#ax2.set_ylabel('mean % error from original')

#plt.figure()

csv_path = "result_smooth_256_30.csv"
linestyle = '.--'

ax = plt.gca()
ax2 = plt.twinx()

df = pandas.read_csv(
    csv_path, usecols=['grid size', 'run time', 'mean_error'])
#grid_min = df[1::]['grid size'].min()
#grid_max = df[1::]['grid size'].max()
#ax.plot(df[1::]['grid size'], df[1::]['run time'], f'b{linestyle}', label=f'runtime s')
#ax.plot(df[1::]['grid size'], df[1::]['mean_error'], f'r{linestyle}', label=f'error %')
#ax = df[1::].plot(x='grid size', y='run time', label=f'runtime s', style=f'b{linestyle}')
#df[1::].plot(x='grid size', label=f'% error', style=f'{linestyle}', subplots=True)

ax.plot(df[1::]['grid size'], df[1::]['run time'], f'b.-', label=f'runtime (s) (blue)')
ax2.plot(df[1::]['grid size'], df[1::]['mean_error'], f'k.--', label=f'error % (black)')
ax2.set_ylabel('mean % error from original')

#ax1.set_ylim(0, 1)
ax.set_ylabel('runtime (s)')
ax.set_xlim(14, 8)  # decreasing time
ax2.set_ylabel('error %')
ax2.set_ylim(0, 1)
#plt.legend()
plt.title('Grid undersampling vs. runtime and error\n(smaller grid is coarser problem)')
#plt.legend()
#ax.legend()
plt.grid()
plt.show()
