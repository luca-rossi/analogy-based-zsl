import json, matplotlib.pyplot as plt

data = dict()
with open("./data/accuracy.json","r") as f:
	data = json.loads(f.read())


def plotConsumptions(dict, type="zsl"):
	labels = []			
	for i in range( len(data.keys())-1 ):
		labels.append(list(data.keys())[i+1])
	colors=["#000000", "#0000FF", "#00FF00", "#FF0066", "#AACC66", "#00AA66"]

	plt.figure()
	if(type == "seen"):
		plt.title("Generalised ZSL - Seen Class")
	elif(type == "unseen"):
		plt.title("Generalised ZSL - Unseen Class")
	else:
		plt.title("Conventional ZSL")

	plt.xlabel('Model')
	plt.ylabel('Accuracy [%]')
	
	
	models = list( data.keys() )
	models.pop(0)
	datasets = list( data[models[0]])
	exp_count = len( list( data[models[0]][datasets[0]]["zsl"] ) )

	#plot ranges		
	yMax, yMin = 0, 1
	for i in range( len(models) ):
		for j in range( len(datasets) ):
			for k in range( exp_count ):
				value = data[models[i]][datasets[j]][type][k]
				if k == 0:
					plt.plot( (i*2+1), value, color=colors[k], marker="o", label="Baseline" )
					plt.text((i*2+1), value, "  "+datasets[j]+" "+str(value), fontsize=8, va='center_baseline', ha='left')	
				else:
					plt.scatter( (i*2+1), value, color=colors[k], marker="o", label="Experiment: "+str(k) )
					plt.text((i*2+1), value, "  "+datasets[j]+" "+str(value), fontsize=8, va='center_baseline', ha='left')	
				#update plot ranges
				yMax = value if value > yMax else yMax
				yMin = value if value!=0 and value < yMin else yMin
	
	plt.gca().legend(('Baseline','1: 2nd most similar', '2: hiddenLayer 2048','5: poolFirst','5.1: poolMax','12: batch 128'), title="Experiments", loc="lower center")
	plt.xticks(ticks=[1,3,5], labels=labels)
	plt.xlim([0, 6])

	yMin -= 0.005
	yMax += 0.005
	plt.ylim( [yMin, yMax] )
	plt.show()

			#[seen, unseen, zsl]
plotConsumptions(data,"zsl")