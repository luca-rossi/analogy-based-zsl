'''Converter dati json per tabella overleaf'''
import json

data = dict()
with open("./data/accuracy.json","r") as f:
	data = json.loads(f.read())


models = list( data.keys() )
models.pop(0)

datasets = list( data[models[0]])
accuracy_types = list(data[models[0]][datasets[0]].keys() )
exp_count = len( list( data[models[0]][datasets[0]][accuracy_types[0]] ) )

#header tabella
table = str('\\'+"begin{table}[!ht] \centering"+ "\\"+"resizebox{1"+"\\textwidth}{!}{"+"\\"+
	    "begin{tabular}{|l|l|ccc|ccc|ccc|ccc|} \hline Model & Experiment & ~ & AWA2 & ~ & ~ & CUB & ~ & ~ & FLO & ~ & ~ & SUN & ~ " +
		"\\"+ "\\"+"hline ~ & ~ & cZSL & Seen & Unseen & cZSL & Seen & Unseen & cZSL & Seen & Unseen & cZSL & Seen & Unseen" + "\\\\")

for i in range( len(models) ):
	table += models[i] + "&\n"
	for k in range( exp_count ):
		#skip first model column
		if not( k == 0 or (i==0 and k==0)) :
			table += " ~ &"
		if k == 0:
			table += "Baseline" + "&\n"
		else:
			table += "n." + str(k) + "&\n"
		for j in range( len(datasets) ):
			for w in accuracy_types:
				if data[models[i]][datasets[j]][w][k] != 0:
					table += str( data[models[i]][datasets[j]][w][k] ) + "&\n"
				else:
					table += " &"
		table += "\n"
	table += "\n"
table += "\\"+"end{tabular}}"+"\n\\"+"end{table}"

with open("./data/accuracy.tex", "w") as f:
    f.write(table)
    f.close