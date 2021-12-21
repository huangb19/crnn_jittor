import json

filename = 'record/6.json'
with open(filename, 'r') as f:
	result = json.load(f)

val = result['val']
val_time = result['val_time']

current_epoch = 0
for i in range(len(val)):
	if val_time[i]['epoch'] != current_epoch:
		if current_epoch != 0:
			print('epoch:', current_epoch)
			print('acc:', val[best]['acc'])
			print('loss:', val[best]['loss'])
			print()
		current_epoch = val_time[i]['epoch']
		best = i
	else:
		if val[i]['acc'] > val[best]['acc']:
			best = i

print('epoch:', current_epoch)
print('acc:', val[best]['acc'])
print('loss:', val[best]['loss'])
print()
