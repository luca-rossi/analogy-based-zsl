import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from codecarbon import EmissionsTracker

tracker = EmissionsTracker( output_dir="./footprint", log_level="error" )
footprints = ["clswgan (regular)", "free (regular)", "tfvaegan (regular)"]

def parseConsumptions(files):
	list = []
	for file in files:	#dataframe
		list.append(pd.read_csv("./footprint/"+file+" footprint.csv"))
	return list


def plotConsumptions(items):
	plt.figure()
	plt.title("Models Footprints  [30 epochs]")
	plt.xlabel('Duration [min]')
	#plt.ylabel('Energy [Wh]')
	plt.xlim([30, 120]) #mins
	plt.ylim([40, 160])

	names, durations, emissions, energy =[],[],[],[]
	for item in items:
		names.append(str(item.project_name[0]))
		durations.append(item.duration[0]/60)
		energy.append(item.energy_consumed[0]*1000)
		emissions.append(item.emissions[0]*1000)

	ax1 = plt.subplot()
	ax1.set_ylabel("Energy [Wh]")
	#ax1.set_ylim(40, 160) #same as main plot

	ax2 = ax1.twinx()
	ax2.set_ylabel("Emissions [g CO2]")
	ax2.set_ylim(-10, 60)

	
	l1 = ax1.plot(durations, energy, color='green', marker="o", label="Energy")
	l2 = ax2.plot(durations, emissions, color='blue', marker="o", label="Emissions")
	for item in range( len(items) ):
		ax1.text(durations[item], energy[item], "  "+names[item], va='top', ha='left')
		ax2.text(durations[item], emissions[item], "  "+names[item], va='top', ha='left' )
	
	ax1.yaxis.label.set_color(l1[0].get_color())
	ax2.yaxis.label.set_color(l2[0].get_color())
	plt.legend(handles=l1+l2, loc='upper left')
	plt.show()


consumptions_list = parseConsumptions(footprints)
plotConsumptions(consumptions_list)