###用於计算和评估预测结果的准确性###

##計算簡單準確率##
#preds預測值 #labels真實標籤
def simple_accuracy(preds, labels):
	cnt = 0
	#用於匹配的數量
	for i in range(len(preds)):
		if preds[i] == labels[i]:
			cnt += 1
			#循環匹配所有預測值和真實值是否一樣，一樣則增加cnt
	return cnt / len(preds)
#return 用於將計算出來準確率的值，返回给調用這個函數的代碼

#更复杂的准确率，特别适用于MBTI这样的多维标签
def full_accuracy(preds, labels):
	# [0 matches, exactly 1 match, exactly 2 matches, exactly 3 matches, exactly 4 matches]
	#记录了不同匹配程度的计数
	num_same_array = [0,0,0,0,0]
	#计算每个维度上的匹配数，并更新 num_same_array
	for i in range(len(preds)):
		curr_cnt = 0
		if (preds[i][0] == labels[i][0]): curr_cnt += 1
		if (preds[i][1] == labels[i][1]): curr_cnt += 1
		if (preds[i][2] == labels[i][2]): curr_cnt += 1
		if (preds[i][3] == labels[i][3]): curr_cnt += 1
		num_same_array[curr_cnt] += 1

	#num_same_array轉換為累計數 #cumulative counts 
	# [0 matches, at least 1 match, at least 2 matches, at least 3 matches, at least 4 matches]
	num_same_array[3] += num_same_array[4]
	num_same_array[2] += num_same_array[3]
	num_same_array[1] += num_same_array[2]

	for i in range(len(num_same_array)):
		num_same_array[i] /= len(preds) #得到比例
		#len(preds)總預測數
	return num_same_array

#單獨計算MBTI每個維度(E/I、N/S、F/T、P/J)的準確率
def mbti_accuracy(preds, labels):
	mbti_accuracy_array = [0, 0, 0, 0]
	for i in range(len(preds)):
		#對於每個預測，如果在維度上匹配，則計數器加1
		if (preds[i][0] == labels[i][0]): mbti_accuracy_array[0] += 1
		if (preds[i][1] == labels[i][1]): mbti_accuracy_array[1] += 1
		if (preds[i][2] == labels[i][2]): mbti_accuracy_array[2] += 1
		if (preds[i][3] == labels[i][3]): mbti_accuracy_array[3] += 1
	for i in range(len(mbti_accuracy_array)):
		mbti_accuracy_array[i] /= len(preds)
		#計算每個維度的準確率
	return mbti_accuracy_array

#把預測和標籤從數字形式轉為相對應的MBTI類型
def convert_to_types(preds, labels, label_list):
	label_map = {}#把數字映射到MBTI類型
	for i in range(len(label_list)):
		label_map[i] = label_list[i]

	new_preds = []
	for i in preds:
		new_preds.append(label_map[i])

	new_labels = []
	for i in labels:
		new_labels.append(label_map[i])
	
	#轉換預測和標籤列表
	return new_preds, new_labels

#計算和評估MBTI類型數據的模型的性能指標
def compute_metrics(preds, labels, label_list):
	#首先把預測和標籤轉為MBTI的類型
	preds, labels = convert_to_types(preds, labels, label_list)
	#計算簡單準確率
	simple_acc = simple_accuracy(preds, labels)
	metrics = {"simple_acc": simple_acc}
	#如果預測結果是單標籤(每個預測只有一個維度)
	if (len(preds[0]) == 1):
		return metrics
	else:
		#為多標籤(MBTI類型)的預測
		#計算至少有1、2、3、4個標籤匹配的情況的準確率
		full_acc = full_accuracy(preds, labels)
		#計算MBTI每個維度的準確率
		mbti_acc = mbti_accuracy(preds, labels)
		#將準確率指標加到metrice中
		metrics["at_least_1_match"] = full_acc[1] 
		metrics["at_least_2_matches"] = full_acc[2] 
		metrics["at_least_3_matches"] = full_acc[3]
		metrics["all_4_matches"] = full_acc[4]
		metrics["E/I"] = mbti_acc[0]
		metrics["N/S"] = mbti_acc[1]
		metrics["F/T"] = mbti_acc[2]
		metrics["P/J"] = mbti_acc[3]
		print(metrics)
		return metrics